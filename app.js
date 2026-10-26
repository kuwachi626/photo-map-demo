/* 学習用: カメラ → 画像分類 → 現在地 → 地図 の順に処理します。 */
'use strict';
const $ = id => document.getElementById(id);
let stream, model, prediction, map, pins, photoReady = false, busy = false;
let records = [], photoVersion = 0, locating = false;
function message(id, text) { $(id).textContent = text; }
function controls() {
  $('classify').disabled = !photoReady || busy;
  $('locate').disabled = !prediction || busy || locating;
  $('load').disabled = busy; $('resetModel').disabled = busy;
}
function stopCamera() {
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  $('video').srcObject = null; $('video').hidden = true;
  $('shoot').disabled = true; $('stop').disabled = true;
  $('photo').hidden = !photoReady; $('placeholder').hidden = photoReady;
}
function newPhoto() {
  photoReady = true; photoVersion++; prediction = null;
  $('photo').hidden = false; $('placeholder').hidden = true;
  message('result', '写真の準備ができました。判定してみよう。'); controls();
}
$('start').onclick = async () => {
  $('start').disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Error('HTTPSまたはlocalhostで開いてください。');
    stopCamera();
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});
    $('video').srcObject = stream; $('video').hidden = false;
    $('photo').hidden = true; $('placeholder').hidden = true;
    await $('video').play(); $('shoot').disabled = false; $('stop').disabled = false;
    message('cameraStatus', 'カメラが起動しました。撮りたいものを写して「撮影する」を押そう。');
  } catch (e) { stopCamera(); message('cameraStatus', 'カメラを起動できません。権限と接続を確認するか、写真を選んでください。 ' + e.message); }
  finally { $('start').disabled = false; }
};
$('stop').onclick = () => { stopCamera(); message('cameraStatus','カメラを停止しました。'); };
$('shoot').onclick = () => {
  const video = $('video'), canvas = $('photo');
  if (!video.videoWidth) return;
  const scale = Math.min(1, 1280 / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
  newPhoto(); stopCamera(); message('cameraStatus','撮影しました。カメラは停止しています。');
};
$('upload').onchange = async e => {
  const file = e.target.files[0]; if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const img = new Image(); img.src = url; await img.decode();
    stopCamera(); const scale = Math.min(1,1280 / Math.max(img.width,img.height));
    const canvas = $('photo'); canvas.width = Math.round(img.width*scale); canvas.height = Math.round(img.height*scale);
    canvas.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height); newPhoto();
    message('cameraStatus','写真を読み込みました。');
  } catch { message('cameraStatus','写真を読み込めません。JPEGやPNGを選んでください。'); }
  finally { URL.revokeObjectURL(url); e.target.value = ''; }
};
$('load').onclick = async () => {
  busy = true; controls(); let candidate;
  try {
    if (!window.tf) throw Error('AIライブラリを読み込めません。ネット接続を確認してください。');
    const files = Array.from($('modelFiles').files); if (!files.length) throw Error('model.jsonと重みの.binを選んでください。');
    message('modelStatus','モデルを読み込み中…');
    candidate = await tf.loadLayersModel(tf.io.browserFiles(files));
    const shape = candidate.inputs[0].shape;
    if (candidate.inputs.length !== 1 || candidate.outputs.length !== 1 || shape.length !== 4 || shape[3] !== 3 || !shape[1] || !shape[2] || ![null,1].includes(shape[0])) throw Error('固定サイズのRGB画像を1枚入力する分類モデルが必要です。');
    if (candidate.outputs[0].shape.length !== 2) throw Error('出力は[1, クラス数]である必要があります。');
    model?.dispose(); model = candidate; candidate = null; prediction = null;
    message('mode','自分たちのAI'); $('demoLabel').hidden = true;
    message('modelStatus',`モデル接続済み：入力 ${shape[2]} × ${shape[1]} px。クラス順と正規化を学習時に合わせてください。`);
    message('result','モデルを接続しました。写真を判定してください。');
  } catch(e) { candidate?.dispose(); message('modelStatus','読み込み失敗：'+e.message); }
  finally { busy = false; controls(); }
};
$('resetModel').onclick = () => {
  model?.dispose(); model = null; prediction = null; $('demoLabel').hidden = false;
  message('mode','体験モード'); message('modelStatus','体験モード：判定の流れを試せます。実際の画像認識ではありません。');
  message('result','体験モードに戻しました。'); controls();
};
$('classify').onclick = async () => {
  busy = true; controls(); const version = photoVersion;
  try {
    let result;
    if (!model) result = {label:$('demoClass').value, demo:true};
    else {
      const labels = $('labels').value.split(',').map(x => x.trim());
      const shape = model.inputs[0].shape;
      const normalization = $('normalization').value, outputMode = $('output').value;
      const tensor = tf.tidy(() => {
        // 中央を正方形に切り出し、モデル入力サイズへ変更します。
        let image = tf.browser.fromPixels($('photo'));
        const [h,w] = image.shape, side = Math.min(h,w);
        image = image.slice([Math.floor((h-side)/2),Math.floor((w-side)/2),0],[side,side,3]);
        let input = tf.image.resizeBilinear(image,[shape[1],shape[2]]).toFloat();
        if (normalization === 'zero') input = input.div(255);
        if (normalization === 'signed') input = input.div(127.5).sub(1);
        const output = model.predict(input.expandDims(0));
        return outputMode === 'logits' ? tf.softmax(output) : output;
      });
      let scores; try { scores = Array.from(await tensor.data()); } finally { tensor.dispose(); }
      if (labels.length !== scores.length || labels.some(x => !x)) throw Error('クラス名の個数をモデルの出力数に合わせてください。');
      const total = scores.reduce((a,b)=>a+b,0);
      if (scores.some(x=>!Number.isFinite(x)||x<0||x>1) || Math.abs(total-1)>0.05) throw Error('出力が確率ではありません。出力設定またはモデルを確認してください。');
      const index = scores.indexOf(Math.max(...scores));
      result = {label:labels[index],score:scores[index],demo:false};
    }
    if (version !== photoVersion) return;
    prediction = result;
    $('result').replaceChildren(); const title = document.createElement('strong'); title.textContent = result.label;
    const note = document.createElement('span'); note.textContent = result.demo ? '体験用の結果（画像認識はしていません）' : `モデルの確信度 ${(result.score*100).toFixed(1)}%${result.score<0.7?' / 確信度が低いため、撮り直してみよう。':''}`;
    $('result').append(title,note);
  } catch(e) { prediction = null; message('result','判定できません：'+e.message); }
  finally { busy = false; controls(); }
};
try {
  if (!window.L) throw Error();
  map = L.map('map').setView([36.2,138.2],5); pins = L.layerGroup().addTo(map);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
} catch { message('map','地図を読み込めません。ネット接続を確認して再読み込みしてください。'); }
$('locate').onclick = () => {
  if (!prediction || !map) { message('geoStatus','地図が読み込まれていません。再読み込みしてください。'); return; }
  if (!navigator.geolocation) { message('geoStatus','このブラウザでは位置情報を利用できません。'); return; }
  locating = true; controls(); message('geoStatus','現在地を取得中…');
  const snapshot = {...prediction, photo:$('photo').toDataURL('image/jpeg',0.7)};
  navigator.geolocation.getCurrentPosition(position => {
    const {latitude:lat,longitude:lng,accuracy} = position.coords;
    const popup = document.createElement('span'); popup.textContent = snapshot.label + (snapshot.demo?'（体験）':'');
    L.marker([lat,lng]).addTo(pins).bindPopup(popup).openPopup();
    L.circle([lat,lng],{radius:accuracy,color:'#477c60',weight:1,fillOpacity:0.08}).addTo(pins);
    map.setView([lat,lng],16); records.unshift(snapshot);
    if (records.length===1) $('records').replaceChildren();
    const row = document.createElement('div'); row.className='record';
    const img = document.createElement('img'); img.src=snapshot.photo; img.alt='記録した写真';
    const text = document.createElement('div'); text.textContent=popup.textContent;
    const small = document.createElement('small'); small.textContent=`${new Date().toLocaleTimeString('ja-JP')} / ${lat.toFixed(5)}, ${lng.toFixed(5)} / 精度 約${Math.round(accuracy)}m`;
    text.append(small); row.append(img,text); $('records').prepend(row);
    message('count',`${records.length} 件の発見`); message('geoStatus','ピンを立てました。円は位置の誤差の目安です。');
    locating=false; controls();
  }, error => {
    const reasons={1:'位置情報の許可が必要です。ブラウザの設定から許可してください。',2:'現在地を取得できません。場所を変えて試してください。',3:'時間がかかりすぎました。もう一度試してください。'};
    message('geoStatus', reasons[error.code] || '位置情報を取得できません。'); locating=false; controls();
  },{enableHighAccuracy:true,timeout:15000,maximumAge:0});
};
$('clear').onclick = () => { if(locating) return; pins?.clearLayers(); records=[]; message('records','発見したものが、ここに並びます。'); message('count','0 件の発見'); message('geoStatus','記録を消しました。'); };
window.addEventListener('pagehide',stopCamera);
