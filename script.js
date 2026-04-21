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
const CHAR_MODEL_PATH = "https://drive.google.com/uc?export=download&id=1HVBKhOucjjgfycmGp2c8Bl13WuD8neXI";
const INPUT_SIZE = 640;

// Ngưỡng cho biển số
const CONF_THRESHOLD_PLATE = 0.5;
const IOU_THRESHOLD_PLATE = 0.45;
const MIN_BOX_SIZE_PLATE = 30;
const MIN_ASPECT_RATIO = 1.5;
const MAX_ASPECT_RATIO = 5.0;

// Ngưỡng cho ký tự
const CONF_THRESHOLD_CHAR = 0.65;
const IOU_THRESHOLD_CHAR = 0.4;
const CHAR_PADDING_RATIO = 0.2;

// FPS giới hạn (2 khung/giây)
const FPS_LIMIT = 2;

// Danh sách class ký tự
const CHAR_CLASSES = {
    0: '-', 
    1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G',
    18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N',
    25: 'O', 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T',
    31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'
};

// ==================== Biến toàn cục ====================
let sessionPlate = null;
let sessionChar = null;
let isModelReady = false;
let isProcessing = false;
let tempCanvas = null;
let tempCtx = null;
let cvReady = false;

let animationId = null;
let lastTimestamp = 0;

// Biến cho zoom thật
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

// ==================== CAMERA VỚI PTZ (ZOOM THẬT) ====================
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
            log("⚠️ Camera không hỗ trợ zoom thật, chỉ có thể zoom ảo (giao diện)", true);
            zoomInfo.innerText = "📱 Camera không hỗ trợ zoom thật";
        }
        log("Camera sau OK");
        return true;
    } catch (err) {
        log("Lỗi camera sau (không có PTZ), thử camera trước", true);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await video.play();
            videoTrack = stream.getVideoTracks()[0];
            log("Camera trước OK (không có zoom thật)");
            zoomInfo.innerText = "📱 Camera không hỗ trợ zoom thật";
            return true;
        } catch (e2) {
            log("Không thể mở camera: " + e2.message, true);
            return false;
        }
    }
}

// Hàm áp dụng zoom thật
async function applyZoom(zoomValue) {
    if (!videoTrack) return;
    const capabilities = videoTrack.getCapabilities();
    if (!capabilities.zoom) return;
    const minZ = capabilities.zoom.min;
    const maxZ = capabilities.zoom.max;
    let clamped = Math.min(Math.max(zoomValue, minZ), maxZ);
    if (Math.abs(clamped - currentZoom) < 0.01) return;
    try {
        await videoTrack.applyConstraints({
            advanced: [{ zoom: clamped }]
        });
        currentZoom = clamped;
        zoomInfo.innerText = `🔍 Zoom: ${currentZoom.toFixed(2)}x (chạm hai ngón)`;
        log(`Zoom thật: ${currentZoom.toFixed(2)}`);
    } catch (err) {
        log("Lỗi áp dụng zoom: " + err.message, true);
    }
}

// Khởi tạo pinch-to-zoom trên video
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
        let newZoom = initialZoom * e.scale;
        applyZoom(newZoom);
    });
    hammerManager.on('pinchend', (e) => {
        e.preventDefault();
        // không cần thêm hành động
    });
    log("✅ Đã kích hoạt pinch-to-zoom trên video");
}

// ==================== Load models ====================
async function loadModel(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
    const buffer = await response.arrayBuffer();
    const sess = await ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"]
    });
    return sess;
}

// ==================== Tiền xử lý ảnh (letterbox) ====================
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

// ==================== XỬ LÝ ẢNH BIỂN SỐ NÂNG CAO (OpenCV) ====================
async function enhancePlateImage(plateCanvas) {
    return new Promise((resolve, reject) => {
        if (!cvReady) {
            reject("OpenCV chưa sẵn sàng");
            return;
        }
        try {
            let src = cv.imread(plateCanvas);
            let gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
            
            let bilateral = new cv.Mat();
            cv.bilateralFilter(gray, bilateral, 9, 75, 75);
            
            let equalized = new cv.Mat();
            cv.equalizeHist(bilateral, equalized);
            
            let kernel = cv.matFromArray(3, 3, cv.CV_32F, [
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0
            ]);
            let sharpened = new cv.Mat();
            cv.filter2D(equalized, sharpened, cv.CV_8U, kernel);
            
            let thresh = new cv.Mat();
            cv.adaptiveThreshold(sharpened, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);
            
            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            let maxArea = 0;
            let bestContour = null;
            for (let i = 0; i < contours.size(); i++) {
                let contour = contours.get(i);
                let area = cv.contourArea(contour);
                if (area > maxArea) {
                    let peri = cv.arcLength(contour, true);
                    let approx = new cv.Mat();
                    cv.approxPolyDP(contour, approx, 0.05 * peri, true);
                    if (approx.rows === 4) {
                        maxArea = area;
                        if (bestContour) bestContour.delete();
                        bestContour = approx.clone();
                    }
                    approx.delete();
                }
            }
            
            let resultMat;
            if (bestContour) {
                let pts = [];
                for (let i = 0; i < 4; i++) {
                    let x = bestContour.data32S[i*2];
                    let y = bestContour.data32S[i*2+1];
                    pts.push({x, y});
                }
                pts.sort((a,b) => a.y - b.y);
                let topPts = pts.slice(0,2).sort((a,b) => a.x - b.x);
                let bottomPts = pts.slice(2,4).sort((a,b) => a.x - b.x);
                let srcPoints = [
                    topPts[0].x, topPts[0].y,
                    topPts[1].x, topPts[1].y,
                    bottomPts[1].x, bottomPts[1].y,
                    bottomPts[0].x, bottomPts[0].y
                ];
                let srcMat = cv.matFromArray(4, 1, cv.CV_32FC2, srcPoints);
                
                let widthTop = Math.hypot(topPts[1].x - topPts[0].x, topPts[1].y - topPts[0].y);
                let widthBottom = Math.hypot(bottomPts[1].x - bottomPts[0].x, bottomPts[1].y - bottomPts[0].y);
                let width = Math.max(widthTop, widthBottom);
                let heightLeft = Math.hypot(bottomPts[0].x - topPts[0].x, bottomPts[0].y - topPts[0].y);
                let heightRight = Math.hypot(bottomPts[1].x - topPts[1].x, bottomPts[1].y - topPts[1].y);
                let height = Math.max(heightLeft, heightRight);
                
                let dstPoints = [0, 0, width, 0, width, height, 0, height];
                let dstMat = cv.matFromArray(4, 1, cv.CV_32FC2, dstPoints);
                
                let perspectiveMat = cv.getPerspectiveTransform(srcMat, dstMat);
                let warped = new cv.Mat();
                cv.warpPerspective(src, warped, perspectiveMat, new cv.Size(width, height));
                resultMat = warped;
                
                srcMat.delete();
                dstMat.delete();
                perspectiveMat.delete();
                bestContour.delete();
            } else {
                resultMat = src.clone();
            }
            
            let outputCanvas = document.createElement("canvas");
            cv.imshow(outputCanvas, resultMat);
            
            src.delete();
            gray.delete();
            bilateral.delete();
            equalized.delete();
            kernel.delete();
            sharpened.delete();
            thresh.delete();
            contours.delete();
            hierarchy.delete();
            if (resultMat) resultMat.delete();
            
            resolve(outputCanvas);
        } catch (err) {
            reject(err);
        }
    });
}

// ==================== Tách dòng & hậu xử lý ====================
function splitLines(boxes) {
    if (boxes.length === 0) return [[], []];
    const sorted = [...boxes].sort((a, b) => a.y1 - b.y1);
    let line1 = [sorted[0]];
    let line2 = [];
    for (let i = 1; i < sorted.length; i++) {
        const prev = line1[line1.length - 1];
        if (Math.abs(sorted[i].y1 - prev.y1) < 20) {
            line1.push(sorted[i]);
        } else {
            line2.push(sorted[i]);
        }
    }
    return [line1, line2];
}

function fixPlateText(text) {
    return text
        .replace(/O/g, '0')
        .replace(/I/g, '1')
        .replace(/Z/g, '2')
        .replace(/S/g, '5')
        .replace(/B/g, '8');
}

// ==================== Nhận diện ký tự trên crop ====================
async function detectCharactersOnCrop(plateCanvas) {
    let enhancedCanvas;
    try {
        enhancedCanvas = await enhancePlateImage(plateCanvas);
    } catch (err) {
        log("OpenCV xử lý lỗi, dùng ảnh gốc: " + err, true);
        enhancedCanvas = plateCanvas;
    }
    
    if (cvReady) {
        try {
            let src = cv.imread(enhancedCanvas);
            let upscaled = new cv.Mat();
            cv.resize(src, upscaled, new cv.Size(0, 0), 2.0, 2.0, cv.INTER_CUBIC);
            let scaledCanvas = document.createElement("canvas");
            cv.imshow(scaledCanvas, upscaled);
            src.delete();
            upscaled.delete();
            enhancedCanvas = scaledCanvas;
        } catch (err) {
            log("Scale up lỗi: " + err, true);
        }
    }
    
    const { tensor, dx, dy, scale } = preprocessImage(enhancedCanvas, enhancedCanvas.width, enhancedCanvas.height, INPUT_SIZE, INPUT_SIZE);
    const results = await sessionChar.run({ [sessionChar.inputNames[0]]: tensor });
    const output = results[sessionChar.outputNames[0]];
    const letterboxInfo = { dx, dy, scale };
    const numBoxesTotal = output.data.length / 36;
    const invScale = 1 / scale;
    
    const charBoxes = [];
    for (let i = 0; i < numBoxesTotal; i++) {
        const cx = output.data[i];
        const cy = output.data[i + numBoxesTotal];
        const w = output.data[i + 2 * numBoxesTotal];
        const h = output.data[i + 3 * numBoxesTotal];
        let bestClass = -1;
        let bestScore = 0;
       const NUM_CLASSES = Object.keys(CHAR_CLASSES).length;
       for (let c = 0; c < NUM_CLASSES; c++) {
            const score = output.data[i + (4 + c) * numBoxesTotal];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestScore < CONF_THRESHOLD_CHAR) continue;
        const label = CHAR_CLASSES[bestClass];
        if (!label) continue;
        
        let x1 = (cx - dx) * invScale - w * invScale / 2;
        let y1 = (cy - dy) * invScale - h * invScale / 2;
        let x2 = (cx - dx) * invScale + w * invScale / 2;
        let y2 = (cy - dy) * invScale + h * invScale / 2;
        x1 = Math.max(0, x1);
        y1 = Math.max(0, y1);
        x2 = Math.min(enhancedCanvas.width, x2);
        y2 = Math.min(enhancedCanvas.height, y2);
        if (x2 - x1 < 5 || y2 - y1 < 5) continue;
        charBoxes.push({ x1, y1, x2, y2, score: bestScore, label });
    }
    
    const nmsChar = nonMaxSuppression(charBoxes.map(b => ({ x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, score: b.score, label: b.label })), IOU_THRESHOLD_CHAR);
    const [lineUp, lineDown] = splitLines(nmsChar);
    lineUp.sort((a,b) => a.x1 - b.x1);
    lineDown.sort((a,b) => a.x1 - b.x1);
    
    let plateText = lineUp.map(b => b.label).join('');
    if (lineDown.length > 0) plateText += lineDown.map(b => b.label).join('');
    plateText = fixPlateText(plateText);
    
    return { boxes: nmsChar, text: plateText };
}

// ==================== Vẽ kết quả ====================
function drawResults(plateBoxes, charResultsForPlate) {
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
    if (charResultsForPlate && charResultsForPlate.boxes) {
        ctx.fillStyle = "#ffaa44";
        ctx.font = "bold 12px monospace";
        for (const ch of charResultsForPlate.boxes) {
            ctx.strokeStyle = "#ffaa44";
            ctx.strokeRect(ch.x1, ch.y1, ch.x2 - ch.x1, ch.y2 - ch.y1);
            ctx.fillText(ch.label, ch.x1 + 2, ch.y1 - 2);
        }
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
        let charData = null;
        if (plateBoxes.length > 0) {
            const bestPlate = plateBoxes[0];
            const cropCanvas = cropImageFromVideo(bestPlate, CHAR_PADDING_RATIO);
            if (cropCanvas) {
                charData = await detectCharactersOnCrop(cropCanvas);
                if (charData.text.length > 0) finalText = `🔢 ${charData.text}`;
                else finalText = `⚠️ Biển số (${Math.round(bestPlate.score*100)}%) nhưng không đọc được ký tự`;
            } else {
                finalText = `⚠️ Lỗi crop biển số`;
            }
        }
        resultTextDiv.innerText = finalText;
        drawResults(plateBoxes, charData);
    } catch (err) {
        console.error(err);
        resultTextDiv.innerText = "⚠️ Lỗi nhận diện";
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
        detect(); // gọi async không await
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
        log("⏳ Đang tải model biển số...");
        sessionPlate = await loadModel(PLATE_MODEL_PATH);
        log("⏳ Đang tải model ký tự...");
        sessionChar = await loadModel(CHAR_MODEL_PATH);
        isModelReady = true;
        initPreprocess();
        
        let waitCount = 0;
        const waitForCV = setInterval(() => {
            if (cvReady) {
                clearInterval(waitForCV);
                log("OpenCV ready, bắt đầu video...");
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
            } else {
                waitCount++;
                if (waitCount > 100) {
                    clearInterval(waitForCV);
                    log("OpenCV không tải được, chạy fallback", true);
                    cvReady = false;
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
                            }
                        }
                    }, 200);
                }
            }
        }, 100);
    } catch (err) {
        log("Lỗi: " + err.message, true);
        startBtn.disabled = false;
        startBtn.innerText = "▶ BẮT ĐẦU QUÉT";
        resultTextDiv.innerText = "⚠️ Lỗi, thử lại";
    }
});
