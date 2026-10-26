# みつけた！ AIフィールドノート

HTML / CSS / JavaScriptだけの学習用デモです。ビルド・APIキー・アプリ用バックエンドは不要です。TensorFlow.js・MobileNet・LeafletはCDNから、学習済みモデルはTensorFlowの配布先から、地図タイルはOpenStreetMapから取得するため、インターネット接続は必要です。モデルは初回に自動で読み込みます。

## はじめに

フォルダー全体をHTTPS対応の静的ホスティングに置き、index.htmlを開いてください。パソコンではlocalhostの静的Webサーバーでも動作します。ファイルを直接ダブルクリックする方法はブラウザによって動作が異なるため推奨しません。スマホで使用する場合はHTTPSのURLで開いてください。

1. 「カメラを起動」を押し、使用を許可します。
2. 「撮影する」を押します。撮影後はカメラが停止します。「写真を選ぶ」でも操作できます。
3. サンプルAI（MobileNet）の読み込み完了後、「この写真を判定する」を押します。実際の画像認識を行い、上位3候補と確信度を表示します。名前は英語です。カメラなしでも「猫のサンプル写真で試す」から判定できます。
4. 「現在地を取得してピンを立てる」を押し、位置情報を許可します。GPSが使えない場合は「サンプル位置（東京駅）で試す」から地図記録を体験できます。こちらは現在地ではなく固定のテスト位置です。

ピンは撮影時の位置ではなく、位置情報ボタンを押した時点の端末の位置です。写真内の被写体の場所を推定する機能ではありません。写真・結果・位置の記録はメモリだけに保存され、ページの再読み込みで消えます。写真とモデルのファイルをサーバーへ送信する処理はありません。地図サービスには地図表示に必要な通信が発生します。

## Colabで学習したモデルを使う

Colabは学習を行う場所です。このデモではColabに写真を送らず、学習済みモデルをブラウザへ読み込んで判定します。標準では学習済みMobileNet V2（alpha=1.0）を自動読み込みします。自分たちのColabモデルに切り替えることもできます。

対応するのはTensorFlow.jsの**Layers形式の画像分類モデル**です。Kerasの.h5 / .keras、SavedModel、.tflite、PyTorchのファイルをそのまま読み込むことはできません。変換に対応したモデルとTensorFlow.jsの互換環境が必要です。

Colabで対応するKerasモデルを変換する例（学習済みモデルの変数名をmodelとする）：

```python
!pip install tensorflowjs
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, 'web_model')
```

書き出したweb_modelフォルダーからmodel.jsonとすべての.binファイルをダウンロードします。変換が失敗する場合は、学習環境・Kerasのバージョン・レイヤーの対応状況を確認してください。

画面の「自分たちのColabモデルを接続」を開き、model.jsonと.binをまとめて選択し、クラス名を学習時の順番で入力します。「モデルを読み込む」を押すと実際のAI判定へ切り替わります。

### モデルとの約束

- 入力は[1, 高さ, 幅, 3]のRGB画像。縦横サイズはモデルから読み取ります。
- 写真の中央を正方形に切り出し、入力サイズにリサイズします。学習時の前処理をこの方法に合わせるか、app.jsの前処理を変更してください。
- 正規化を0〜1、−1〜1、0〜255から選択できます。モデル内に正規化レイヤーがある場合は二重に正規化しないようにしてください。
- 出力は[1, クラス数]。確率なら「確率」、未正規化スコアなら「スコア」を選択します。独立した複数ラベルや二値分類の単一出力には対応していません。
- クラス名の順序は、学習コードのclass_names / class_indicesなどで確認します。
- 確信度はモデルの出力で、正解率や正しさの保証ではありません。

## 授業で見せるコード

- index.html：画面の構成
- style.css：色やレイアウト
- app.js：カメラの起動、撮影、モデルの読み込み、判定、位置情報、ピンの追加

公式資料：
- https://www.tensorflow.org/js/guide/save_load
- https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- https://developer.mozilla.org/en-US/docs/Web/API/Geolocation/getCurrentPosition

## 検証状況

JavaScriptの構文とローカルHTTP配信を確認済み。ブラウザでMobileNetの読み込み・猫写真の実際の画像判定・サンプル位置へのピン表示を確認済みです。カメラ起動・撮影・停止、位置情報成功と拒否、記録消去、モデル読込失敗の処理は模擬APIで検証しました。実カメラは未確認です。検証環境のGPS取得はタイムアウトしたため、実際の現在地は利用する端末で確認してください。

## サンプルモデルと素材

- モデル：TensorFlow.js MobileNet V2。ImageNetの1000クラスを分類します。未知の物でも既知の候補を返すため、万能な物体認識・植物図鑑ではありません。確信度が低いときは撮り直すか別の物で試してください。
- モデル説明：https://github.com/tensorflow/tfjs-models/tree/master/mobilenet
- 猫画像：TensorFlow公式サンプル https://github.com/tensorflow/tfjs-examples/blob/master/mobilenet/cat.jpg
- サンプル素材のライセンスはassets/TENSORFLOW-LICENSE.txtを参照。
