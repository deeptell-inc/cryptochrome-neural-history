"""Read published XLSX values without reconstructing trials or filling missing cells."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
BOOK = ROOT / '41586_2025_8734_MOESM4_ESM.xlsx'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def extract():
    sheets = {}
    with zipfile.ZipFile(BOOK) as archive:
        strings = [''.join(e.itertext()) for e in ET.fromstring(
            archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        rel = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(
            archive.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            target = rel[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            cells = {}
            for cell in ET.fromstring(archive.read(target)).findall('.//s:sheetData/s:row/s:c', NS):
                value = cell.find('s:v', NS)
                if value is not None:
                    cells[cell.attrib['r']] = strings[int(value.text)] if cell.attrib.get('t') == 's' else value.text
                elif cell.attrib.get('t') == 'inlineStr':
                    cells[cell.attrib['r']] = ''.join(cell.find('s:is', NS).itertext())
            sheets[sheet.attrib['name']] = cells
    return sheets


if __name__ == '__main__':
    sheets = extract()
    (ROOT / 'source_data_cells.json').write_text(json.dumps(sheets, indent=2, ensure_ascii=False))
    cells = sheets['Fig. 4f']
    assert cells['C1'] == 't fast (ms)' and cells['H1'] == 't slow (ms)'
    assert cells['C3'] == '0' and cells['D3'] == '10'
    rows = sorted(int(k[1:]) for k, v in cells.items()
                  if re.fullmatch(r'A\d+', k) and v.startswith('Cell '))
    summary = {
        'doi': '10.1038/s41586-025-08734-4',
        'source_sha256': hashlib.sha256(BOOK.read_bytes()).hexdigest(),
        'sheet': 'Fig. 4f', 'cell_system': 'adult Drosophila dFBN',
        'intervention': '50 uM 4-ONE in patch intracellular solution; holding -80 mV except test protocols',
        'replicate_unit': 'published cell row; not repeated raw sweeps',
        'uncertainty': 'sample SD across cells, not channel noise or SEM',
        'raw_current_traces_available_in_workbook': False,
        'results': {},
    }
    for label, cols in [('tau_fast_ms', ['C', 'D']), ('tau_slow_ms', ['H', 'I'])]:
        pairs = [(r, float(cells[f'{cols[0]}{r}']), float(cells[f'{cols[1]}{r}']))
                 for r in rows if all(f'{c}{r}' in cells for c in cols)]
        assert len(pairs) == 11
        before, after = ([p[i] for p in pairs] for i in [1, 2])
        delta = [b-a for a, b in zip(before, after)]
        summary['results'][label] = {
            'n_paired_cells': len(pairs),
            'before_mean': statistics.mean(before), 'before_sample_sd': statistics.stdev(before),
            'after_mean': statistics.mean(after), 'after_sample_sd': statistics.stdev(after),
            'paired_change_mean': statistics.mean(delta), 'paired_change_sample_sd': statistics.stdev(delta),
            'source_cells': [[f'{c}{r}' for c in cols] for r, _, _ in pairs],
        }
    (ROOT / 'source_data_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['results'], indent=2))
