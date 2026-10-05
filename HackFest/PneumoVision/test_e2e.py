import requests, json

BASE_URL = 'http://127.0.0.1:8000'

print('=== 1. Testing GET / ===')
r = requests.get(BASE_URL + '/')
print('Status:', r.status_code)
assert 'PneumoVision' in r.text
assert 'Automated Dual-Engine Pipeline' in r.text
assert '19 Models' not in r.text
assert 'threshold-slider' not in r.text
print('GET / verified: Clean, modern UI served.')

print('\n=== 2. Testing Static Assets ===')
r_css = requests.get(BASE_URL + '/static/style.css')
print('style.css status:', r_css.status_code, 'bytes:', len(r_css.content))
r_js = requests.get(BASE_URL + '/static/script.js')
print('script.js status:', r_js.status_code, 'bytes:', len(r_js.content))
assert r_css.status_code == 200
assert r_js.status_code == 200

print('\n=== 3. Testing GET /health ===')
r_health = requests.get(BASE_URL + '/health').json()
print('Health:', json.dumps(r_health, indent=2))
assert r_health['status'] == 'healthy'
assert r_health['model_loaded'] is True
assert r_health['classification_loaded'] is True

print('\n=== 4. Testing GET /models ===')
r_models = requests.get(BASE_URL + '/models').json()
print('Primary detection:', r_models['primary_detection']['name'])
print('Curated classifier:', r_models['curated_classification']['name'])

print('\n=== 5. Testing GET /sample_images ===')
r_samples = requests.get(BASE_URL + '/sample_images').json()
print('Samples count:', len(r_samples['samples']))
for s in r_samples['samples']:
    print(' -', s['filename'], f"({round(s['size_bytes']/1024, 1)} KB)", s['type'])

print('\n=== 6. Testing POST /predict (DICOM Sample) ===')
with open('sample_images/sample_chest_radiograph.dcm', 'rb') as fp:
    r_pred = requests.post(BASE_URL + '/predict', files={'file': fp})
print('Predict DICOM status:', r_pred.status_code)
pred_data = r_pred.json()
print('Finding:', pred_data['finding'])
print('Latency:', pred_data['inference_time_ms'], 'ms')
print('Classification:', pred_data['classification'])

print('\n=== 7. Testing POST /report/download (HTML & Text) ===')
r_html_rep = requests.post(BASE_URL + '/report/download', json={**pred_data, 'format': 'html'})
print('HTML Report status:', r_html_rep.status_code, 'bytes:', len(r_html_rep.content))
assert 'PneumoVision' in r_html_rep.text
assert 'Faster R-CNN' in r_html_rep.text
assert 'DenseNet-121' in r_html_rep.text

r_txt_rep = requests.post(BASE_URL + '/report/download', json={**pred_data, 'format': 'text'})
print('Text Report status:', r_txt_rep.status_code, 'bytes:', len(r_txt_rep.content))
assert 'PNEUMOVISION' in r_txt_rep.text
assert 'DenseNet-121' in r_txt_rep.text

print('\n========================================')
print('ALL 7 SYSTEM VERIFICATION CHECKS PASSED!')
print('========================================')
