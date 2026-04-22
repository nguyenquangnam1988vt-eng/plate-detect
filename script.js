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
const RECOGNIZER_PATH = "/model/english_g2_jpqd.onnx";
const INPUT_SIZE = 640;

// Ngưỡng cho biển số
const CONF_THRESHOLD_PLATE = 0.5;
const IOU_THRESHOLD_PLATE = 0.45;
const MIN_BOX_SIZE_PLATE = 30;
const MIN_ASPECT_RATIO = 1.5;
const MAX_ASPECT_RATIO = 5.0;
const CHAR_PADDING_RATIO = 0.2;

// FPS giới hạn
const FPS_LIMIT = 3; // tăng lên 3 để mượt hơn

// Bảng ký tự chuẩn cho biển số (số và chữ in hoa)
const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const BLANK_INDEX = 0; // blank là index 0 (theo model)

// ==================== Biến toàn cục ====================
let sessionPlate = null;
let sessionRecognizer = null;
let isModelReady = false;
let isProcessing = false;
let tempCanvas = null;
let tempCtx = null;
let cvReady = false;

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

// ==================== OpenCV ready callback ====================
window._cvReadyCallback = function() {
    cvReady = true;
    log("OpenCV.js đã sẵn sàng");
};

// ==================== CAMERA VỚI ZOOM THẬT ====================
async function startCamera() {
    try {
        log("📷 Mở camera sau (yêu cầu zoom)...");
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
            log(`✅ Camera hỗ trợ zoom: min=${capabilities.zoom.min}, max=${capabilities.zoom.max}`);
            currentZoom = capabilities.zoom.min || 1;
            zoomInfo.innerText = `🔍 Hỗ trợ zoom (chạm hai ngón) - Hiện tại: ${currentZoom.toFixed(2)}x`;
            initPinchToZoom();
        } else {
            log("⚠️ Camera không hỗ trợ zoom thật", true);
            zoomInfo.innerText = "📱 Camera không hỗ trợ zoom thật";
        }
        return true;
    } catch (err) {
        log("Lỗi camera sau, thử camera trước: " + err.message, true);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await video.play();
            videoTrack = stream.getVideoTracks()[0];
            log("Camera trước OK (không zoom thật)");
            zoomInfo.innerText = "📱 Camera không hỗ trợ zoom thật";
            return true;
        } catch (e2) {
            log("Không thể mở camera: " + e2.message, true);
            return false;
        }
    }
}

async function applyZoom(zoomValue) {
    if (!videoTrack) return;
    const caps = videoTrack.getCapabilities();
    if (!caps.zoom) return;
    const minZ = caps.zoom.min;
    const maxZ = caps.zoom.max;
    let clamped = Math.min(Math.max(zoomValue, minZ), maxZ);
    if (Math.abs(clamped - currentZoom) < 0.01) return;
    try {
        await videoTrack.applyConstraints({ advanced: [{ zoom: clamped }] });
        currentZoom = clamped;
        zoomInfo.innerText = `🔍 Zoom: ${currentZoom.toFixed(2)}x (chạm hai ngón)`;
        log(`Zoom thật: ${currentZoom.toFixed(2)}`);
    } catch (err) {
        log("Lỗi zoom: " + err.message, true);
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
    hammerManager.on('pinchstart', (e) => {
        e.preventDefault();
        initialZoom = currentZoom;
    });
    hammerManager.on('pinchmove', (e) => {
        e.preventDefault();
        applyZoom(initialZoom * e.scale);
    });
    log("✅ Pinch-to-zoom sẵn sàng");
}

// ==================== Load models ====================
async function loadModel(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
    const buffer = await response.arrayBuffer();
    const sess = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
    return sess;
}

// ==================== Tiền xử lý cho YOLO (biển số) ====================
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
        input[i] = imgData[i * 4] / 255.0;
        input[i + outWidth * outHeight] = imgData[i * 4 + 1] / 255.0;
        input[i + 2 * outWidth * outHeight] = imgData[i * 4 + 2] / 255.0;
    }
    return { tensor: new ort.Tensor("float32", input, [1, 3, outWidth, outHeight]), dx, dy, scale };
}

// ==================== Parse YOLO output ====================
function parseYoloOutput(outputData, dims, imgW, imgH, numAttrsExpected, letterboxInfo, confThresh, minBoxSize, minAspect, maxAspect) {
    let numBoxes, numAttrs;
    if (dims.length === 3) {
        if (dims[1] === numAttrsExpected && dims[2] > 1000) {
            numAttrs = dims[1];
            numBoxes = dims[2];
        } else if (dims[2] === numAttrsExpected && dims[1] > 1000) {
            numAttrs = dims[2];
            numBoxes = dims[1];
        } else {
            numAttrs = numAttrsExpected;
            numBoxes = outputData.length / numAttrsExpected;
        }
    } else {
        numAttrs = numAttrsExpected;
        numBoxes = outputData.length / numAttrsExpected;
    }
    const numClasses = numAttrs - 4;
    const { dx, dy, scale } = letterboxInfo;
    const invScale = 1 / scale;

    const boxes = [];
    for (let i = 0; i < numBoxes; i++) {
        const cx = outputData[i];
        const cy = outputData[i + numBoxes];
        const w = outputData[i + 2 * numBoxes];
        const h = outputData[i + 3 * numBoxes];
        let conf = 0;
        if (numClasses === 1) {
            conf = outputData[i + 4 * numBoxes];
        } else {
            for (let c = 0; c < numClasses; c++) {
                const score = outputData[i + (4 + c) * numBoxes];
                if (score > conf) conf = score;
            }
        }
        if (conf < confThresh) continue;

        let x1 = (cx - dx) * invScale - w * invScale / 2;
        let y1 = (cy - dy) * invScale - h * invScale / 2;
        let x2 = (cx - dx) * invScale + w * invScale / 2;
        let y2 = (cy - dy) * invScale + h * invScale / 2;

        x1 = Math.max(0, x1);
        y1 = Math.max(0, y1);
        x2 = Math.min(imgW, x2);
        y2 = Math.min(imgH, y2);

        const boxW = x2 - x1;
        const boxH = y2 - y1;
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
    const union = areaA + areaB - inter;
    return inter / (union + 1e-6);
}

function nonMaxSuppression(boxes, iouThresh) {
    if (!boxes.length) return [];
    boxes.sort((a,b) => b.score - a.score);
    const result = [];
    for (let i = 0; i < boxes.length; i++) {
        let keep = true;
        for (let j = 0; j < result.length; j++) {
            if (iou(boxes[i], result[j]) > iouThresh) {
                keep = false;
                break;
            }
        }
        if (keep) result.push(boxes[i]);
    }
    return result;
}

// ==================== Crop ảnh từ video ====================
function cropImageFromVideo(box, paddingRatio) {
    const padX = (box.x2 - box.x1) * paddingRatio;
    const padY = (box.y2 - box.y1) * paddingRatio;
    let cropX1 = Math.max(0, box.x1 - padX);
    let cropY1 = Math.max(0, box.y1 - padY);
    let cropX2 = Math.min(video.videoWidth, box.x2 + padX);
    let cropY2 = Math.min(video.videoHeight, box.y2 + padY);
    const cropW = cropX2 - cropX1;
    const cropH = cropY2 - cropY1;
    if (cropW <= 0 || cropH <= 0) return null;

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.drawImage(video, cropX1, cropY1, cropW, cropH, 0, 0, cropW, cropH);
    return cropCanvas;
}

// ==================== NHẬN DIỆN KÝ TỰ DÙNG CRNN (EasyOCR) ====================
// Hàm reshape output của model (xử lý [1,T,C] hoặc [1,C,T])
function reshapeOutput(output) {
    const dims = output.dims; // [1, T, C] hoặc [1, C, T]
    const data = output.data;
    let T, C;
    let logits = [];

    if (dims[1] > dims[2]) {
        // dạng [1, C, T]
        C = dims[1];
        T = dims[2];
        for (let t = 0; t < T; t++) {
            let row = [];
            for (let c = 0; c < C; c++) {
                row.push(data[c * T + t]);
            }
            logits.push(row);
        }
    } else {
        // dạng [1, T, C]
        T = dims[1];
        C = dims[2];
        for (let t = 0; t < T; t++) {
            let row = [];
            for (let c = 0; c < C; c++) {
                row.push(data[t * C + c]);
            }
            logits.push(row);
        }
    }
    return logits;
}

// Tiền xử lý ảnh cho CRNN: giữ tỷ lệ, padding đen, resize về 32x100
function preprocessForCRNN(canvas) {
    const targetW = 100;
    const targetH = 32;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = targetW;
    tempCanvas.height = targetH;
    const ctx = tempCanvas.getContext("2d");

    const scale = Math.min(targetW / canvas.width, targetH / canvas.height);
    const newW = canvas.width * scale;
    const newH = canvas.height * scale;
    const dx = (targetW - newW) / 2;
    const dy = (targetH - newH) / 2;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, targetW, targetH);
    ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, dx, dy, newW, newH);

    const imgData = ctx.getImageData(0, 0, targetW, targetH).data;

    const input = new Float32Array(targetW * targetH);
    for (let i = 0; i < targetW * targetH; i++) {
        const gray = 0.299 * imgData[i*4] + 0.587 * imgData[i*4+1] + 0.114 * imgData[i*4+2];
        input[i] = gray / 255.0;
    }

    return new ort.Tensor("float32", input, [1, 1, targetH, targetW]);
}

// Giải mã CTC (greedy)
function ctcGreedyDecode(logits2D, blankIdx) {
    if (!logits2D.length) return "";
    const T = logits2D.length;
    let prevIdx = blankIdx;
    let resultIndices = [];
    for (let t = 0; t < T; t++) {
        let maxIdx = 0;
        let maxVal = logits2D[t][0];
        for (let c = 1; c < logits2D[t].length; c++) {
            if (logits2D[t][c] > maxVal) {
                maxVal = logits2D[t][c];
                maxIdx = c;
            }
        }
        if (maxIdx !== blankIdx && maxIdx !== prevIdx) {
            resultIndices.push(maxIdx);
        }
        prevIdx = maxIdx;
    }
    // Chuyển index sang ký tự (lưu ý: CHARSET bắt đầu từ index 0, blank ở index 0)
    // Nếu model có blank ở index 0 thì các ký tự thật bắt đầu từ index 1.
    // Ta map: index-1 vào CHARSET, nếu index=0 thì bỏ qua.
    let text = "";
    for (let idx of resultIndices) {
        if (idx > 0 && idx-1 < CHARSET.length) {
            text += CHARSET[idx-1];
        }
    }
    return text;
}

// Lọc và format biển số
function cleanPlate(text) {
    // Chỉ giữ chữ in hoa và số
    let cleaned = text.toUpperCase().replace(/[^A-Z0-9]/g, '');
    // Thêm dấu gạch ngang đơn giản (3 số đầu, dấu gạch, các số còn lại)
    // Nếu độ dài >= 3: ví dụ "51A12345" -> "51A-12345"
    if (cleaned.length >= 3) {
        // Tìm vị trí cắt: có thể cắt sau 3 ký tự hoặc sau chữ cái cuối cùng của phần đầu
        // Ở đây cắt sau 3 ký tự đầu tiên
        cleaned = cleaned.slice(0,3) + "-" + cleaned.slice(3);
    }
    return cleaned;
}

async function recognizePlateText(plateCanvas) {
    try {
        const inputTensor = preprocessForCRNN(plateCanvas);
        const results = await sessionRecognizer.run({ [sessionRecognizer.inputNames[0]]: inputTensor });
        const output = results[sessionRecognizer.outputNames[0]];
        const logits2D = reshapeOutput(output);
        const rawText = ctcGreedyDecode(logits2D, BLANK_INDEX);
        let finalText = cleanPlate(rawText);
        return finalText;
    } catch (err) {
        log("Lỗi nhận diện ký tự: " + err.message, true);
        return "";
    }
}

// ==================== Vẽ kết quả ====================
function drawResults(plateBoxes, recognizedText) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#00ff9c";
    ctx.lineWidth = 3;
    for (const box of plateBoxes) {
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        ctx.fillStyle = "#00ff9c";
        ctx.font = "bold 14px monospace";
        ctx.fillText(`${Math.round(box.score * 100)}%`, box.x1 + 4, box.y1 - 4);
    }
    if (recognizedText && plateBoxes.length > 0) {
        ctx.fillStyle = "#ffaa44";
        ctx.font = "bold 18px monospace";
        ctx.shadowColor = "black";
        ctx.shadowBlur = 2;
        ctx.fillText(recognizedText, plateBoxes[0].x1, plateBoxes[0].y1 - 10);
        ctx.shadowColor = "transparent";
    }
}

// ==================== Luồng chính detect ====================
async function detect() {
    if (!isModelReady || isProcessing) return;
    if (!video.videoWidth || video.videoWidth === 0) return;
    isProcessing = true;
    try {
        const { tensor, dx, dy, scale } = preprocessImage(video, video.videoWidth, video.videoHeight, INPUT_SIZE, INPUT_SIZE);
        const resultsPlate = await sessionPlate.run({ [sessionPlate.inputNames[0]]: tensor });
        const outputPlate = resultsPlate[sessionPlate.outputNames[0]];
        let plateBoxes = parseYoloOutput(outputPlate.data, outputPlate.dims, video.videoWidth, video.videoHeight, 5, { dx, dy, scale }, CONF_THRESHOLD_PLATE, MIN_BOX_SIZE_PLATE, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO);
        plateBoxes = nonMaxSuppression(plateBoxes, IOU_THRESHOLD_PLATE);
        
        let finalText = "🚫 Không thấy biển số";
        let recognizedText = "";
        
        if (plateBoxes.length > 0) {
            // Chọn box có score cao nhất
            const bestPlate = plateBoxes.reduce((a, b) => a.score > b.score ? a : b);
            const cropCanvas = cropImageFromVideo(bestPlate, CHAR_PADDING_RATIO);
            if (cropCanvas) {
                recognizedText = await recognizePlateText(cropCanvas);
                if (recognizedText.length > 0) {
                    finalText = `🔢 ${recognizedText}`;
                } else {
                    finalText = `⚠️ Biển số (${Math.round(bestPlate.score*100)}%) nhưng không đọc được ký tự`;
                }
            } else {
                finalText = `⚠️ Lỗi crop biển số`;
            }
        }
        resultTextDiv.innerText = finalText;
        drawResults(plateBoxes, recognizedText);
    } catch (err) {
        console.error(err);
        resultTextDiv.innerText = "⚠️ Lỗi nhận diện";
        log("Detect error: " + err.message, true);
    } finally {
        isProcessing = false;
    }
}

// Vòng lặp giới hạn FPS
function detectLoop(now) {
    if (!lastTimestamp) lastTimestamp = now;
    const delta = now - lastTimestamp;
    if (delta >= 1000 / FPS_LIMIT && !isProcessing) {
        lastTimestamp = now;
        detect();
    }
    animationId = requestAnimationFrame(detectLoop);
}

function startDetectionLoop() {
    if (animationId) cancelAnimationFrame(animationId);
    lastTimestamp = 0;
    animationId = requestAnimationFrame(detectLoop);
    log(`🔄 Vòng quét với FPS_LIMIT = ${FPS_LIMIT} (giảm tải CPU)`);
}

// ==================== KHỞI TẠO ====================
startBtn.addEventListener("click", async () => {
    if (startBtn.disabled) return;
    startBtn.disabled = true;
    startBtn.innerText = "⏳ ĐANG KHỞI TẠO...";
    resultTextDiv.innerText = "📷 Khởi động...";
    try {
        log("🚀 Bắt đầu");
        const cameraOK = await startCamera();
        if (!cameraOK) throw new Error("Camera lỗi");
        
        log("⏳ Đang tải model biển số (bienso1.onnx)...");
        sessionPlate = await loadModel(PLATE_MODEL_PATH);
        log("⏳ Đang tải model nhận dạng ký tự (english_g2_jpqd.onnx)...");
        sessionRecognizer = await loadModel(RECOGNIZER_PATH);
        isModelReady = true;
        initPreprocess();
        
        let attempts = 0;
        const waitForVideo = setInterval(() => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                clearInterval(waitForVideo);
                log(`Video ready: ${video.videoWidth}x${video.videoHeight}`);
                startDetectionLoop();
                startBtn.innerText = "🔄 ĐANG QUÉT";
                resultTextDiv.innerText = "🔍 Đang quan sát...";
            } else {
                attempts++;
                if (attempts > 50) {
                    clearInterval(waitForVideo);
                    log("Video timeout", true);
                    startBtn.disabled = false;
                    startBtn.innerText = "▶ BẮT ĐẦU QUÉT";
                } else if (attempts % 10 === 0) {
                    log(`⏳ Đợi video... (${attempts}/50)`);
                }
            }
        }, 200);
    } catch (err) {
        log("Lỗi: " + err.message, true);
        startBtn.disabled = false;
        startBtn.innerText = "▶ BẮT ĐẦU QUÉT";
        resultTextDiv.innerText = "⚠️ Lỗi, thử lại";
    }
});
