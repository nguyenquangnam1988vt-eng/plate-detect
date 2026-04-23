<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Nhận diện biển số xe Việt Nam - OCR + Threshold</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0f1e; font-family: 'Segoe UI', sans-serif; padding: 16px; color: white; }
        .container { max-width: 700px; margin: 0 auto; }
        .video-wrapper { position: relative; width: 100%; background: #000; border-radius: 20px; overflow: hidden; margin-bottom: 16px; }
        video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: auto; border-radius: 20px; }
        video { position: relative; z-index: 1; }
        canvas { z-index: 2; }
        .result-panel { background: #1e2a3a; border-radius: 20px; padding: 16px; margin-top: 20px; text-align: center; }
        #resultText { font-size: 28px; font-weight: bold; background: #000000aa; padding: 12px; border-radius: 50px; font-family: monospace; }
        button { background: #00b4d8; border: none; padding: 14px 28px; font-size: 18px; font-weight: bold; border-radius: 50px; color: white; margin: 10px 0; width: 100%; cursor: pointer; }
        button:disabled { background: #555; cursor: not-allowed; }
        #zoomInfo { font-size: 14px; background: #00000099; display: inline-block; padding: 4px 12px; border-radius: 20px; }
        #logBox { background: #000000cc; font-size: 11px; height: 150px; overflow-y: auto; padding: 8px; border-radius: 12px; margin-top: 12px; font-family: monospace; }
        .status { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
    <script async src="https://docs.opencv.org/4.8.0/opencv.js" onload="window._cvReadyCallback()"></script>
</head>
<body>
<div class="container">
    <div class="video-wrapper" style="position: relative; padding-bottom: 75%;">
        <video id="video" autoplay playsinline muted style="position: absolute; width:100%; height:100%; object-fit: cover;"></video>
        <canvas id="canvas" style="position: absolute; width:100%; height:100%;"></canvas>
    </div>
    <div class="status">
        <span id="zoomInfo">🔍 Zoom: --</span>
        <button id="startBtn">▶ BẮT ĐẦU QUÉT</button>
    </div>
    <div class="result-panel">
        <div id="resultText">🚫 Chưa nhận diện</div>
    </div>
    <div id="logBox"></div>
</div>

<script>
// ==================== DOM elements ====================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const resultTextDiv = document.getElementById("resultText");
const startBtn = document.getElementById("startBtn");
const logBox = document.getElementById("logBox");
const zoomInfo = document.getElementById("zoomInfo");
const ctx = canvas.getContext("2d");

// ==================== CẤU HÌNH ====================
const PLATE_MODEL_PATH = "/model/bienso1.onnx";
const OCR_REC_PATH = "/model/en_PP-OCRv3_rec_infer.onnx";
// CLS vẫn load nhưng sẽ không dùng (có thể comment nếu muốn)
const OCR_CLS_PATH = "/model/ch_ppocr_mobile_v2.0_cls_infer.onnx";
const INPUT_SIZE = 640;

const CONF_THRESHOLD_PLATE = 0.5;
const IOU_THRESHOLD_PLATE = 0.45;
const MIN_BOX_SIZE_PLATE = 30;
const MIN_ASPECT_RATIO_PLATE = 1.5;
const MAX_ASPECT_RATIO_PLATE = 5.0;
const FPS_LIMIT = 5;

// ==================== Biến toàn cục ====================
let sessionPlate = null, sessionOCR = null, sessionCLS = null;
let isModelReady = false, isProcessing = false, cvReady = false;
let tempCanvas = null, tempCtx = null;
let animationId = null, lastTimestamp = 0;
let videoTrack = null, currentZoom = 1, hammerManager = null;

// ==================== Log ====================
function log(message, isError = false) {
    const prefix = isError ? "❌ " : "✅ ";
    const formatted = prefix + message;
    console.log(formatted);
    if (logBox) {
        logBox.innerText += formatted + "\n";
        logBox.scrollTop = logBox.scrollHeight;
    }
}

window._cvReadyCallback = () => { cvReady = true; log("OpenCV.js sẵn sàng"); };

// ==================== Camera & zoom ====================
async function startCamera() {
    try {
        log("📷 Mở camera sau...");
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", zoom: true } });
        video.srcObject = stream;
        video.setAttribute("playsinline", "");
        video.muted = true;
        await video.play();
        videoTrack = stream.getVideoTracks()[0];
        const caps = videoTrack.getCapabilities();
        if (caps.zoom) {
            log(`✅ Zoom: min=${caps.zoom.min}, max=${caps.zoom.max}`);
            currentZoom = caps.zoom.min || 1;
            zoomInfo.innerText = `🔍 Zoom: ${currentZoom.toFixed(2)}x (chạm hai ngón)`;
            initPinchToZoom();
        } else {
            log("⚠️ Camera không hỗ trợ zoom thật", true);
            zoomInfo.innerText = "📱 Không zoom thật";
        }
        return true;
    } catch (err) {
        log("Lỗi camera: " + err.message, true);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await video.play();
            videoTrack = stream.getVideoTracks()[0];
            log("Camera trước OK");
            zoomInfo.innerText = "📱 Không zoom thật";
            return true;
        } catch { return false; }
    }
}

async function applyZoom(zoomValue) {
    if (!videoTrack) return;
    const caps = videoTrack.getCapabilities();
    if (!caps.zoom) return;
    let clamped = Math.min(Math.max(zoomValue, caps.zoom.min), caps.zoom.max);
    if (Math.abs(clamped - currentZoom) < 0.01) return;
    try {
        await videoTrack.applyConstraints({ advanced: [{ zoom: clamped }] });
        currentZoom = clamped;
        zoomInfo.innerText = `🔍 Zoom: ${currentZoom.toFixed(2)}x`;
    } catch (err) { log("Zoom error: " + err.message, true); }
}

function initPinchToZoom() {
    if (!video) return;
    if (hammerManager) hammerManager.destroy();
    hammerManager = new Hammer.Manager(video);
    const pinch = new Hammer.Pinch();
    hammerManager.add(pinch);
    pinch.set({ enable: true });
    let initialZoom = 1;
    hammerManager.on('pinchstart', (e) => { e.preventDefault(); initialZoom = currentZoom; });
    hammerManager.on('pinchmove', (e) => { e.preventDefault(); applyZoom(initialZoom * e.scale); });
    log("✅ Pinch-to-zoom ready");
}

async function loadModel(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
    const buffer = await response.arrayBuffer();
    return await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
}

// ==================== YOLO Preprocess ====================
function initPreprocess() {
    if (!tempCanvas) {
        tempCanvas = document.createElement("canvas");
        tempCanvas.width = INPUT_SIZE;
        tempCanvas.height = INPUT_SIZE;
        tempCtx = tempCanvas.getContext("2d");
    }
}

function preprocessImage(source, srcWidth, srcHeight, outWidth, outHeight) {
    const scale = Math.min(outWidth / srcWidth, outHeight / srcHeight);
    const newW = srcWidth * scale, newH = srcHeight * scale;
    const dx = (outWidth - newW) / 2, dy = (outHeight - newH) / 2;
    tempCtx.fillStyle = "black";
    tempCtx.fillRect(0, 0, outWidth, outHeight);
    tempCtx.drawImage(source, dx, dy, newW, newH);
    const imgData = tempCtx.getImageData(0, 0, outWidth, outHeight).data;
    const input = new Float32Array(3 * outWidth * outHeight);
    for (let i = 0; i < outWidth * outHeight; i++) {
        input[i] = imgData[i*4] / 255.0;
        input[i + outWidth*outHeight] = imgData[i*4+1] / 255.0;
        input[i + 2*outWidth*outHeight] = imgData[i*4+2] / 255.0;
    }
    return { tensor: new ort.Tensor("float32", input, [1, 3, outWidth, outHeight]), dx, dy, scale };
}

function parseYoloOutput(outputData, dims, imgW, imgH, numAttrsExpected, letterboxInfo, confThresh, minBoxSize, minAspect, maxAspect) {
    let numBoxes, numAttrs;
    if (dims.length === 3) {
        if (dims[1] === numAttrsExpected && dims[2] > 1000) { numAttrs = dims[1]; numBoxes = dims[2]; }
        else if (dims[2] === numAttrsExpected && dims[1] > 1000) { numAttrs = dims[2]; numBoxes = dims[1]; }
        else { numAttrs = numAttrsExpected; numBoxes = outputData.length / numAttrsExpected; }
    } else { numAttrs = numAttrsExpected; numBoxes = outputData.length / numAttrsExpected; }
    const numClasses = numAttrs - 4;
    const { dx, dy, scale } = letterboxInfo;
    const invScale = 1 / scale;
    const boxes = [];
    for (let i = 0; i < numBoxes; i++) {
        const cx = outputData[i], cy = outputData[i + numBoxes];
        const w = outputData[i + 2*numBoxes], h = outputData[i + 3*numBoxes];
        let conf = 0;
        if (numClasses === 1) conf = outputData[i + 4*numBoxes];
        else for (let c = 0; c < numClasses; c++) conf = Math.max(conf, outputData[i + (4+c)*numBoxes]);
        if (conf < confThresh) continue;
        let x1 = (cx - dx)*invScale - w*invScale/2, y1 = (cy - dy)*invScale - h*invScale/2;
        let x2 = (cx - dx)*invScale + w*invScale/2, y2 = (cy - dy)*invScale + h*invScale/2;
        x1 = Math.max(0, x1); y1 = Math.max(0, y1);
        x2 = Math.min(imgW, x2); y2 = Math.min(imgH, y2);
        const boxW = x2 - x1, boxH = y2 - y1;
        if (boxW < minBoxSize || boxH < minBoxSize) continue;
        const aspect = boxW / boxH;
        if (aspect < minAspect || aspect > maxAspect) continue;
        boxes.push({ x1, y1, x2, y2, score: conf });
    }
    return boxes;
}

function iou(boxA, boxB) {
    const x1 = Math.max(boxA.x1, boxB.x1), y1 = Math.max(boxA.y1, boxB.y1);
    const x2 = Math.min(boxA.x2, boxB.x2), y2 = Math.min(boxA.y2, boxB.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
    const areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);
    return inter / (areaA + areaB - inter + 1e-6);
}

function nonMaxSuppression(boxes, iouThresh) {
    if (!boxes.length) return [];
    boxes.sort((a,b) => b.score - a.score);
    const result = [];
    for (let i = 0; i < boxes.length; i++) {
        let keep = true;
        for (let j = 0; j < result.length; j++) {
            if (iou(boxes[i], result[j]) > iouThresh) { keep = false; break; }
        }
        if (keep) result.push(boxes[i]);
    }
    return result;
}

// ==================== TĂNG CƯỜNG + THRESHOLD ====================
function enhancePlateContrast(plateCanvas) {
    return new Promise((resolve) => {
        if (!cvReady) { log("OpenCV chưa ready, bỏ qua contrast", true); resolve(plateCanvas); return; }
        try {
            let src = cv.imread(plateCanvas);
            let gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            // CLAHE
            let clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
            let equalized = new cv.Mat();
            clahe.apply(gray, equalized);
            // Gaussian blur
            let blurred = new cv.Mat();
            cv.GaussianBlur(equalized, blurred, new cv.Size(3, 3), 0);
            // Sharpening
            let kernel = cv.matFromArray(3, 3, cv.CV_32F, [0, -1, 0, -1, 5, -1, 0, -1, 0]);
            let sharpened = new cv.Mat();
            cv.filter2D(blurred, sharpened, -1, kernel);
            // Chuyển về 8-bit
            let resultU8 = new cv.Mat();
            sharpened.convertTo(resultU8, cv.CV_8U);
            
            // THÊM OTSU THRESHOLD: nhị phân hóa, chữ trắng nền đen
            let binary = new cv.Mat();
            cv.threshold(resultU8, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
            
            const outCanvas = document.createElement("canvas");
            outCanvas.width = plateCanvas.width;
            outCanvas.height = plateCanvas.height;
            cv.imshow(outCanvas, binary);
            
            src.delete(); gray.delete(); equalized.delete(); blurred.delete();
            kernel.delete(); sharpened.delete(); resultU8.delete(); binary.delete();
            resolve(outCanvas);
        } catch (err) { log("Contrast error: " + err.message, true); resolve(plateCanvas); }
    });
}

function upscalePlate(canvas, scale = 3) {
    const up = document.createElement("canvas");
    up.width = canvas.width * scale;
    up.height = canvas.height * scale;
    const uctx = up.getContext("2d");
    uctx.imageSmoothingEnabled = true;
    uctx.drawImage(canvas, 0, 0, up.width, up.height);
    log(`📐 Upscale ${scale}x: ${canvas.width}x${canvas.height} → ${up.width}x${up.height}`);
    return up;
}

// ==================== OCR PREPROCESS (STRETCH) ====================
function preprocessPlateForOCR(canvas) {
    const targetW = 320;
    const targetH = 48;
    const temp = document.createElement("canvas");
    temp.width = targetW;
    temp.height = targetH;
    const tctx = temp.getContext("2d");
    tctx.fillStyle = "black";
    tctx.fillRect(0, 0, targetW, targetH);
    // STRETCH: kéo dãn ảnh vừa khít khung, không giữ tỷ lệ
    tctx.drawImage(canvas, 0, 0, targetW, targetH);
    
    const imgData = tctx.getImageData(0, 0, targetW, targetH).data;
    const input = new Float32Array(3 * targetW * targetH);
    for (let i = 0; i < targetW * targetH; i++) {
        input[i] = imgData[i*4] / 255;
        input[i + targetW*targetH] = imgData[i*4+1] / 255;
        input[i + 2*targetW*targetH] = imgData[i*4+2] / 255;
    }
    return new ort.Tensor("float32", input, [1, 3, targetH, targetW]);
}

function reshapeCRNNOutput(output) {
    const dims = output.dims;
    const data = output.data;
    let T, C, logits = [];
    if (dims[1] > dims[2]) { C = dims[1]; T = dims[2]; for (let t=0; t<T; t++) { let row = []; for (let c=0; c<C; c++) row.push(data[c*T + t]); logits.push(row); } }
    else { T = dims[1]; C = dims[2]; for (let t=0; t<T; t++) { let row = []; for (let c=0; c<C; c++) row.push(data[t*C + c]); logits.push(row); } }
    return logits;
}

function ctcGreedyDecode(logits2D, blankIdx) {
    if (!logits2D.length) return [];
    let prevIdx = blankIdx;
    let result = [];
    for (let t=0; t<logits2D.length; t++) {
        let maxIdx = 0, maxVal = logits2D[t][0];
        for (let c=1; c<logits2D[t].length; c++) {
            if (logits2D[t][c] > maxVal) { maxVal = logits2D[t][c]; maxIdx = c; }
        }
        if (maxIdx !== blankIdx && maxIdx !== prevIdx) result.push(maxIdx);
        prevIdx = maxIdx;
    }
    return result;
}

function decodeTextFromIndices(indices, charset) {
    let text = "";
    for (let idx of indices) {
        if (idx > 0 && idx-1 < charset.length) text += charset[idx-1];
    }
    return text;
}

function formatVietnamPlate(raw) {
    let text = raw.toUpperCase();
    text = text.replace(/O/g, "0").replace(/I/g, "1").replace(/B/g, "8").replace(/Q/g, "0").replace(/S/g, "5").replace(/Z/g, "2");
    text = text.replace(/[^A-Z0-9]/g, '');
    let match = text.match(/^(\d{2})([A-Z]{1,2})(\d{4,5})$/);
    if (match) {
        let province = match[1], letters = match[2], numbers = match[3];
        if (numbers.length === 5) return `${province}${letters}-${numbers.slice(0,3)}.${numbers.slice(3)}`;
        else return `${province}${letters}-${numbers}`;
    }
    return text;
}

async function recognizePlateFull(plateCanvas) {
    try {
        log("🔍 Bắt đầu OCR...");
        let enhanced = await enhancePlateContrast(plateCanvas);
        enhanced = upscalePlate(enhanced, 3);  // upscale 3x
        
        // === BỎ CLS: dùng luôn ảnh sau upscale ===
        const inputTensor = preprocessPlateForOCR(enhanced);
        
        const results = await sessionOCR.run({ [sessionOCR.inputNames[0]]: inputTensor });
        const output = results[sessionOCR.outputNames[0]];
        const logits2D = reshapeCRNNOutput(output);
        const indices = ctcGreedyDecode(logits2D, 0);
        const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let rawText = decodeTextFromIndices(indices, CHARSET);
        log(`📝 Raw OCR: "${rawText}"`);
        const formatted = formatVietnamPlate(rawText);
        log(`🏁 Kết quả: "${formatted}"`);
        return formatted;
    } catch (err) {
        log("OCR error: " + err.message, true);
        return "";
    }
}

// ==================== VẼ KẾT QUẢ ====================
function drawResults(plateBoxes, recognizedText) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#00ff9c";
    ctx.lineWidth = 3;
    for (const box of plateBoxes) {
        ctx.strokeRect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1);
        ctx.fillStyle = "#00ff9c";
        ctx.font = "bold 14px monospace";
        ctx.fillText(`${Math.round(box.score*100)}%`, box.x1+4, box.y1-4);
    }
    if (recognizedText && plateBoxes.length) {
        ctx.fillStyle = "#ffaa44";
        ctx.font = "bold 20px monospace";
        ctx.shadowColor = "black";
        ctx.shadowBlur = 2;
        ctx.fillText(recognizedText, plateBoxes[0].x1, plateBoxes[0].y1-10);
        ctx.shadowColor = "transparent";
    }
}

// ==================== DETECT LOOP ====================
async function detect() {
    if (!isModelReady || isProcessing) return;
    if (!video.videoWidth) return;
    isProcessing = true;
    try {
        const { tensor, dx, dy, scale } = preprocessImage(video, video.videoWidth, video.videoHeight, INPUT_SIZE, INPUT_SIZE);
        const plateResults = await sessionPlate.run({ [sessionPlate.inputNames[0]]: tensor });
        const outputPlate = plateResults[sessionPlate.outputNames[0]];
        let plateBoxes = parseYoloOutput(outputPlate.data, outputPlate.dims, video.videoWidth, video.videoHeight, 5, {dx,dy,scale}, CONF_THRESHOLD_PLATE, MIN_BOX_SIZE_PLATE, MIN_ASPECT_RATIO_PLATE, MAX_ASPECT_RATIO_PLATE);
        plateBoxes = nonMaxSuppression(plateBoxes, IOU_THRESHOLD_PLATE);
        
        let finalText = "🚫 Không thấy biển số";
        let recognizedPlateText = "";
        if (plateBoxes.length) {
            const bestPlate = plateBoxes.reduce((a,b) => a.score > b.score ? a : b);
            const padRatio = 0.12;
            const padX = (bestPlate.x2 - bestPlate.x1) * padRatio;
            const padY = (bestPlate.y2 - bestPlate.y1) * padRatio;
            const cropX1 = Math.max(0, bestPlate.x1 - padX);
            const cropY1 = Math.max(0, bestPlate.y1 - padY);
            const cropX2 = Math.min(video.videoWidth, bestPlate.x2 + padX);
            const cropY2 = Math.min(video.videoHeight, bestPlate.y2 + padY);
            const plateCanvas = document.createElement("canvas");
            plateCanvas.width = cropX2 - cropX1;
            plateCanvas.height = cropY2 - cropY1;
            const pctx = plateCanvas.getContext("2d");
            pctx.drawImage(video, cropX1, cropY1, plateCanvas.width, plateCanvas.height, 0, 0, plateCanvas.width, plateCanvas.height);
            
            recognizedPlateText = await recognizePlateFull(plateCanvas);
            if (recognizedPlateText && recognizedPlateText.length >= 5) finalText = `🔢 ${recognizedPlateText}`;
            else finalText = `⚠️ Biển số (${Math.round(bestPlate.score*100)}%) không đọc được`;
        }
        resultTextDiv.innerText = finalText;
        drawResults(plateBoxes, recognizedPlateText);
    } catch(err) {
        console.error(err);
        resultTextDiv.innerText = "⚠️ Lỗi nhận diện";
        log("Detect error: " + err.message, true);
    } finally {
        isProcessing = false;
    }
}

function detectLoop(now) {
    if (!lastTimestamp) lastTimestamp = now;
    if (now - lastTimestamp >= 1000/FPS_LIMIT && !isProcessing) {
        lastTimestamp = now;
        detect();
    }
    animationId = requestAnimationFrame(detectLoop);
}

function startDetectionLoop() {
    if (animationId) cancelAnimationFrame(animationId);
    lastTimestamp = 0;
    animationId = requestAnimationFrame(detectLoop);
}

// ==================== KHỞI TẠO ====================
startBtn.addEventListener("click", async () => {
    if (startBtn.disabled) return;
    startBtn.disabled = true;
    startBtn.innerText = "⏳ ĐANG KHỞI TẠO...";
    try {
        log("🚀 Bắt đầu");
        const cameraOK = await startCamera();
        if (!cameraOK) throw new Error("Camera lỗi");
        log("⏳ Tải model YOLO plate...");
        sessionPlate = await loadModel(PLATE_MODEL_PATH);
        log("⏳ Tải model OCR recognition...");
        sessionOCR = await loadModel(OCR_REC_PATH);
        // Vẫn load CLS nhưng không dùng (có thể bỏ nếu muốn tiết kiệm thời gian)
        // sessionCLS = await loadModel(OCR_CLS_PATH);
        isModelReady = true;
        initPreprocess();
        let attempts = 0;
        const waitVideo = setInterval(() => {
            if (video.videoWidth > 0) {
                clearInterval(waitVideo);
                log(`Video ready: ${video.videoWidth}x${video.videoHeight}`);
                startDetectionLoop();
                startBtn.innerText = "🔄 ĐANG QUÉT";
                resultTextDiv.innerText = "🔍 Đang quan sát...";
            } else if (++attempts > 50) {
                clearInterval(waitVideo);
                log("Video timeout", true);
                startBtn.disabled = false;
                startBtn.innerText = "▶ BẮT ĐẦU QUÉT";
            }
        }, 200);
    } catch(err) {
        log("Lỗi: " + err.message, true);
        startBtn.disabled = false;
        startBtn.innerText = "▶ BẮT ĐẦU QUÉT";
        resultTextDiv.innerText = "⚠️ Lỗi, thử lại";
    }
});
</script>
</body>
</html>
