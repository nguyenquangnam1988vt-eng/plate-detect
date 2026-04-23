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
const OCR_CLS_PATH = "/model/ch_ppocr_mobile_v2.0_cls_infer.onnx";
const INPUT_SIZE = 640; // YOLO dùng 640x640

// YOLO plate params
const CONF_THRESHOLD_PLATE = 0.5;
const IOU_THRESHOLD_PLATE = 0.45;
const MIN_BOX_SIZE_PLATE = 30;
const MIN_ASPECT_RATIO_PLATE = 1.5;
const MAX_ASPECT_RATIO_PLATE = 5.0;

// FPS
const FPS_LIMIT = 5;

// ==================== Biến toàn cục ====================
let sessionPlate = null;
let sessionOCR = null;
let sessionCLS = null; // có thể dùng để chỉnh hướng ảnh sau
let isModelReady = false;
let isProcessing = false;
let cvReady = false;

let tempCanvas = null;
let tempCtx = null;

let animationId = null;
let lastTimestamp = 0;

let videoTrack = null;
let currentZoom = 1;
let hammerManager = null;

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

// ==================== OpenCV ready ====================
window._cvReadyCallback = function() {
    cvReady = true;
    log("OpenCV.js sẵn sàng");
};

// ==================== Camera & zoom ====================
async function startCamera() {
    try {
        log("📷 Mở camera sau...");
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", pan: true, tilt: true, zoom: true }
        });
        video.srcObject = stream;
        video.setAttribute("playsinline", "");
        video.muted = true;
        await video.play();

        videoTrack = stream.getVideoTracks()[0];
        const capabilities = videoTrack.getCapabilities();
        if (capabilities.zoom) {
            log(`✅ Zoom: min=${capabilities.zoom.min}, max=${capabilities.zoom.max}`);
            currentZoom = capabilities.zoom.min || 1;
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
        } catch (e2) {
            log("Không thể mở camera", true);
            return false;
        }
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
    } catch (err) {
        log("Zoom error: " + err.message, true);
    }
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

// ==================== Load models ====================
async function loadModel(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
    const buffer = await response.arrayBuffer();
    return await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
}

// ==================== Preprocess cho YOLO (giữ nguyên) ====================
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
    const newW = srcWidth * scale;
    const newH = srcHeight * scale;
    const dx = (outWidth - newW) / 2;
    const dy = (outHeight - newH) / 2;
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

// ==================== Parse YOLO output (plate detection) ====================
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
        const cx = outputData[i];
        const cy = outputData[i + numBoxes];
        const w = outputData[i + 2*numBoxes];
        const h = outputData[i + 3*numBoxes];
        let conf = 0;
        if (numClasses === 1) conf = outputData[i + 4*numBoxes];
        else for (let c = 0; c < numClasses; c++) conf = Math.max(conf, outputData[i + (4+c)*numBoxes]);
        if (conf < confThresh) continue;
        let x1 = (cx - dx)*invScale - w*invScale/2;
        let y1 = (cy - dy)*invScale - h*invScale/2;
        let x2 = (cx - dx)*invScale + w*invScale/2;
        let y2 = (cy - dy)*invScale + h*invScale/2;
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
    const x1 = Math.max(boxA.x1, boxB.x1);
    const y1 = Math.max(boxA.y1, boxB.y1);
    const x2 = Math.min(boxA.x2, boxB.x2);
    const y2 = Math.min(boxA.y2, boxB.y2);
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

// ==================== PPOCR RECOGNITION (thay thế hoàn toàn CRAFT + CRNN cũ) ====================
function preprocessPlateForOCR(canvas) {
    // Target height = 48 (PPOCRv3 rec thường dùng 48)
    const targetH = 48;
    const ratio = canvas.width / canvas.height;
    let targetW = Math.floor(targetH * ratio);
    // Giới hạn chiều rộng tối đa 320 để tránh tensor quá lớn
    targetW = Math.min(320, targetW);
    if (targetW < 1) targetW = 1;

    const temp = document.createElement("canvas");
    temp.width = targetW;
    temp.height = targetH;
    const tctx = temp.getContext("2d");

    // Nền đen (có thể thay bằng padding giữ tỷ lệ)
    tctx.fillStyle = "black";
    tctx.fillRect(0, 0, targetW, targetH);
    // Vẽ ảnh crop vào giữa (giữ tỷ lệ)
    const drawW = targetW;
    const drawH = targetH;
    tctx.drawImage(canvas, 0, 0, drawW, drawH);

    const imgData = tctx.getImageData(0, 0, targetW, targetH).data;
    const input = new Float32Array(3 * targetW * targetH);
    for (let i = 0; i < targetW * targetH; i++) {
        input[i] = imgData[i*4] / 255.0;         // R
        input[i + targetW*targetH] = imgData[i*4+1] / 255.0; // G
        input[i + 2*targetW*targetH] = imgData[i*4+2] / 255.0; // B
    }
    return new ort.Tensor("float32", input, [1, 3, targetH, targetW]);
}

// Hàm reshape CRNN output (giữ nguyên từ code cũ, dùng cho OCR)
function reshapeCRNNOutput(output) {
    const dims = output.dims;
    const data = output.data;
    let T, C, logits = [];
    if (dims[1] > dims[2]) {
        C = dims[1]; T = dims[2];
        for (let t=0; t<T; t++) {
            let row = [];
            for (let c=0; c<C; c++) row.push(data[c*T + t]);
            logits.push(row);
        }
    } else {
        T = dims[1]; C = dims[2];
        for (let t=0; t<T; t++) {
            let row = [];
            for (let c=0; c<C; c++) row.push(data[t*C + c]);
            logits.push(row);
        }
    }
    return logits;
}

function ctcGreedyDecode(logits2D, blankIdx) {
    if (!logits2D.length) return "";
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
        if (idx >= 0 && idx < charset.length) text += charset[idx];
    }
    return text;
}

// Hàm fix biển số Việt Nam theo quy tắc thực tế
function fixPlateVN(text) {
    // Thay thế các ký tự dễ nhầm
    text = text.replace(/O/g, "0");
    text = text.replace(/I/g, "1");
    text = text.replace(/B/g, "8");
    // Loại bỏ ký tự đặc biệt, chỉ giữ A-Z0-9
    text = text.replace(/[^A-Z0-9]/g, '');
    // Format: 2 số + chữ + 4-5 số
    const match = text.match(/^(\d{2})([A-Z])(\d{4,5})$/);
    if (match) {
        let [, p, l, n] = match;
        // Nếu 5 số thì tách 3 và 2, nếu 4 số thì tách 2 và 2 (hoặc 3.1 tùy) - theo chuẩn hay dùng 3.2
        if (n.length === 5) {
            return `${p}${l}-${n.slice(0,3)}.${n.slice(3)}`;
        } else if (n.length === 4) {
            return `${p}${l}-${n.slice(0,2)}.${n.slice(2)}`;
        }
        return `${p}${l}-${n}`;
    }
    return text;
}

async function recognizePlateFull(plateCanvas) {
    try {
        // Tiền xử lý ảnh biển số
        const inputTensor = preprocessPlateForOCR(plateCanvas);
        // Chạy OCR recognition model
        const results = await sessionOCR.run({ [sessionOCR.inputNames[0]]: inputTensor });
        const output = results[sessionOCR.outputNames[0]];
        const logits2D = reshapeCRNNOutput(output);
        const indices = ctcGreedyDecode(logits2D, 0); // blank index = 0
        // Bộ charset của model PPOCRv3 (thường là 36 ký tự: digits + uppercase, không có blank)
        // Ở đây ta dùng CHARSET gốc từ code cũ (đã khớp với model)
        const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let rawText = decodeTextFromIndices(indices, CHARSET);
        rawText = rawText.toUpperCase();
        rawText = rawText.replace(/[^A-Z0-9]/g, '');
        // Áp dụng fix cho biển số Việt Nam
        const fixed = fixPlateVN(rawText);
        return fixed;
    } catch (err) {
        log("OCR error: " + err.message, true);
        return "";
    }
}

// ==================== Vẽ kết quả ====================
function drawResults(plateBoxes, recognizedText) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    // Vẽ box biển số
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

// ==================== Luồng chính detect (đã sửa) ====================
async function detect() {
    if (!isModelReady || isProcessing) return;
    if (!video.videoWidth) return;
    isProcessing = true;
    try {
        // 1. Detect plate with YOLO
        const { tensor, dx, dy, scale } = preprocessImage(video, video.videoWidth, video.videoHeight, INPUT_SIZE, INPUT_SIZE);
        const plateResults = await sessionPlate.run({ [sessionPlate.inputNames[0]]: tensor });
        const outputPlate = plateResults[sessionPlate.outputNames[0]];
        let plateBoxes = parseYoloOutput(outputPlate.data, outputPlate.dims, video.videoWidth, video.videoHeight, 5, {dx,dy,scale}, CONF_THRESHOLD_PLATE, MIN_BOX_SIZE_PLATE, MIN_ASPECT_RATIO_PLATE, MAX_ASPECT_RATIO_PLATE);
        plateBoxes = nonMaxSuppression(plateBoxes, IOU_THRESHOLD_PLATE);
        
        let finalText = "🚫 Không thấy biển số";
        let recognizedPlateText = "";
        if (plateBoxes.length) {
            const bestPlate = plateBoxes.reduce((a,b) => a.score > b.score ? a : b);
            // Crop biển số từ video (có thể thêm padding 10-15% để tránh mất ký tự)
            const padRatio = 0.12; // 12% padding
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
            
            // Gọi OCR full biển số thay vì CRAFT + CRNN từng ký tự
            recognizedPlateText = await recognizePlateFull(plateCanvas);
            if (recognizedPlateText) finalText = `🔢 ${recognizedPlateText}`;
            else finalText = `⚠️ Biển số (${Math.round(bestPlate.score*100)}%) nhưng không đọc được ký tự`;
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

// ==================== FPS loop ====================
function detectLoop(now) {
    if (!lastTimestamp) lastTimestamp = now;
    if (now - lastTimestamp >= 1000/FPS_LIMIT && !isProcessing) {
        lastTimestamp = now;
        detect();
    }
    animationId = requestAnimationFrame(detectLoop);
}

// ==================== Khởi tạo ====================
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
        log("⏳ Tải model orientation classifier (optional)...");
        sessionCLS = await loadModel(OCR_CLS_PATH);
        isModelReady = true;
        initPreprocess();
        // Chờ video ready
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

function startDetectionLoop() {
    if (animationId) cancelAnimationFrame(animationId);
    lastTimestamp = 0;
    animationId = requestAnimationFrame(detectLoop);
}
