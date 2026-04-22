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
const CRAFT_MODEL_PATH = "/model/craft_mlt_25k_jpqd.onnx";
const RECOGNIZER_PATH = "/model/english_g2_jpqd.onnx";
const INPUT_SIZE = 640; // YOLO và CRAFT đều dùng 640x640

// YOLO plate params
const CONF_THRESHOLD_PLATE = 0.5;
const IOU_THRESHOLD_PLATE = 0.45;
const MIN_BOX_SIZE_PLATE = 30;
const MIN_ASPECT_RATIO_PLATE = 1.5;
const MAX_ASPECT_RATIO_PLATE = 5.0;

// CRAFT character detection params
const CRAFT_CONF_THRESHOLD = 0.4;      // ngưỡng region map
const CRAFT_IOU_THRESHOLD = 0.3;       // NMS cho các box ký tự
const MIN_CHAR_SIZE = 8;               // pixel tối thiểu
const MAX_CHAR_ASPECT = 1.5;           // ký tự thường không quá rộng

// CRNN recognition params
const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const BLANK_INDEX = 0;
const CHAR_PADDING_RATIO = 0.1;        // padding khi crop từ biển số

// FPS
const FPS_LIMIT = 5;

// ==================== Biến toàn cục ====================
let sessionPlate = null;
let sessionCraft = null;
let sessionRecognizer = null;
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

// ==================== Preprocess cho YOLO và CRAFT (giống nhau) ====================
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

// ==================== CRAFT: detect character boxes on plate crop ====================
async function detectCharacterBoxesWithCRAFT(plateCanvas) {
    if (!cvReady) {
        log("OpenCV chưa ready, skip CRAFT", true);
        return [];
    }
    try {
        // Preprocess plateCanvas to 640x640
        const { tensor, dx, dy, scale } = preprocessImage(plateCanvas, plateCanvas.width, plateCanvas.height, INPUT_SIZE, INPUT_SIZE);
        const results = await sessionCraft.run({ [sessionCraft.inputNames[0]]: tensor });
        const output = results[sessionCraft.outputNames[0]];
        // output shape: [1, 2, 640, 640]? Thực tế CRAFT thường trả về region map và affinity map.
        // Ta lấy channel đầu tiên (region map)
        let regionMap;
        if (output.dims.length === 4 && output.dims[1] >= 1) {
            const C = output.dims[1];
            const H = output.dims[2];
            const W = output.dims[3];
            // Lấy channel 0
            regionMap = new Float32Array(H * W);
            for (let i = 0; i < H * W; i++) {
                regionMap[i] = output.data[i]; // vì data layout [batch, channel, h, w] liên tiếp
            }
            // Resize region map về kích thước gốc của plateCanvas
            let srcMat = cv.matFromArray(H, W, cv.CV_32F, regionMap);
            let dstMat = new cv.Mat();
            cv.resize(srcMat, dstMat, new cv.Size(plateCanvas.width, plateCanvas.height), 0, 0, cv.INTER_LINEAR);
            const regionData = new Float32Array(plateCanvas.width * plateCanvas.height);
            for (let i = 0; i < plateCanvas.width * plateCanvas.height; i++) {
                regionData[i] = dstMat.data[i];
            }
            srcMat.delete(); dstMat.delete();
            // Tìm contours
            let binary = cv.matFromArray(plateCanvas.height, plateCanvas.width, cv.CV_32F, regionData);
            let binaryU8 = new cv.Mat();
            cv.threshold(binary, binaryU8, CRAFT_CONF_THRESHOLD, 255, cv.THRESH_BINARY);
            binaryU8.convertTo(binaryU8, cv.CV_8U);
            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.findContours(binaryU8, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            const boxes = [];
            for (let i = 0; i < contours.size(); i++) {
                let contour = contours.get(i);
                let rect = cv.boundingRect(contour);
                let w = rect.width, h = rect.height;
                if (w < MIN_CHAR_SIZE || h < MIN_CHAR_SIZE) continue;
                let aspect = w / h;
                if (aspect > MAX_CHAR_ASPECT) continue;
                boxes.push({
                    x1: rect.x,
                    y1: rect.y,
                    x2: rect.x + rect.width,
                    y2: rect.y + rect.height,
                    score: 1.0
                });
            }
            binary.delete(); binaryU8.delete(); contours.delete(); hierarchy.delete();
            // NMS
            const nmsBoxes = nonMaxSuppression(boxes, CRAFT_IOU_THRESHOLD);
            // Sắp xếp từ trái sang phải, trên xuống dưới
            nmsBoxes.sort((a,b) => {
                if (Math.abs(a.y1 - b.y1) < 10) return a.x1 - b.x1;
                return a.y1 - b.y1;
            });
            return nmsBoxes;
        } else {
            log("CRAFT output shape unexpected", true);
            return [];
        }
    } catch (err) {
        log("CRAFT detection error: " + err.message, true);
        return [];
    }
}

// ==================== Tiền xử lý nâng cao cho crop ký tự ====================
function preprocessCharImage(charCanvas) {
    return new Promise((resolve) => {
        if (!cvReady) {
            // Fallback: chỉ resize giữ tỷ lệ
            const targetW = 100, targetH = 32;
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = targetW;
            tempCanvas.height = targetH;
            const tctx = tempCanvas.getContext("2d");
            const scale = Math.min(targetW / charCanvas.width, targetH / charCanvas.height);
            const newW = charCanvas.width * scale;
            const newH = charCanvas.height * scale;
            const dx = (targetW - newW)/2, dy = (targetH - newH)/2;
            tctx.fillStyle = "black";
            tctx.fillRect(0,0,targetW,targetH);
            tctx.drawImage(charCanvas, dx, dy, newW, newH);
            const imgData = tctx.getImageData(0,0,targetW,targetH).data;
            const input = new Float32Array(targetW * targetH);
            for (let i=0; i<targetW*targetH; i++) {
                const gray = 0.299*imgData[i*4] + 0.587*imgData[i*4+1] + 0.114*imgData[i*4+2];
                input[i] = gray/255.0;
            }
            resolve(new ort.Tensor("float32", input, [1,1,targetH,targetW]));
            return;
        }
        // OpenCV advanced preprocessing
        try {
            let src = cv.imread(charCanvas);
            let gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            // CLAHE
            let clahe = new cv.CLAHE(2.0, new cv.Size(8,8));
            let equalized = new cv.Mat();
            clahe.apply(gray, equalized);
            // Gaussian blur
            let blurred = new cv.Mat();
            cv.GaussianBlur(equalized, blurred, new cv.Size(3,3), 0);
            // Sharpening
            let kernel = cv.matFromArray(3,3, cv.CV_32F, [0,-1,0, -1,5,-1, 0,-1,0]);
            let sharpened = new cv.Mat();
            cv.filter2D(blurred, sharpened, -1, kernel);
            // Adaptive threshold
            let binary = new cv.Mat();
            cv.adaptiveThreshold(sharpened, binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);
            // Resize to 100x32 với padding giữ tỷ lệ
            const targetW = 100, targetH = 32;
            let resized = new cv.Mat();
            let resizedFloat = new cv.Mat();
            binary.convertTo(resizedFloat, cv.CV_32F, 1/255.0);
            cv.resize(resizedFloat, resized, new cv.Size(targetW, targetH), 0, 0, cv.INTER_LINEAR);
            // Chuyển sang array
            let inputData = new Float32Array(targetW * targetH);
            for (let i=0; i<targetW*targetH; i++) {
                inputData[i] = resized.data[i];
            }
            // Cleanup
            src.delete(); gray.delete(); equalized.delete(); blurred.delete();
            kernel.delete(); sharpened.delete(); binary.delete(); resizedFloat.delete(); resized.delete();
            resolve(new ort.Tensor("float32", inputData, [1,1,targetH,targetW]));
        } catch(err) {
            log("OpenCV preprocess error: " + err.message, true);
            // Fallback
            const targetW = 100, targetH = 32;
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = targetW;
            tempCanvas.height = targetH;
            const tctx = tempCanvas.getContext("2d");
            tctx.drawImage(charCanvas, 0, 0, charCanvas.width, charCanvas.height, 0, 0, targetW, targetH);
            const imgData = tctx.getImageData(0,0,targetW,targetH).data;
            const input = new Float32Array(targetW * targetH);
            for (let i=0; i<targetW*targetH; i++) {
                const gray = 0.299*imgData[i*4] + 0.587*imgData[i*4+1] + 0.114*imgData[i*4+2];
                input[i] = gray/255.0;
            }
            resolve(new ort.Tensor("float32", input, [1,1,targetH,targetW]));
        }
    });
}

// ==================== CRNN recognition ====================
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
    let text = "";
    for (let idx of result) {
        if (idx > 0 && idx-1 < CHARSET.length) text += CHARSET[idx-1];
    }
    return text;
}

async function recognizeCharFromCanvas(charCanvas) {
    try {
        const inputTensor = await preprocessCharImage(charCanvas);
        const results = await sessionRecognizer.run({ [sessionRecognizer.inputNames[0]]: inputTensor });
        const output = results[sessionRecognizer.outputNames[0]];
        const logits2D = reshapeCRNNOutput(output);
        let rawText = ctcGreedyDecode(logits2D, BLANK_INDEX);
        rawText = rawText.toUpperCase();
        // Lọc chỉ A-Z,0-9
        rawText = rawText.replace(/[^A-Z0-9]/g, '');
        return rawText.length === 1 ? rawText : '';
    } catch(err) {
        log("Recognize char error: " + err.message, true);
        return '';
    }
}

// ==================== Crop ký tự từ biển số ====================
function cropCharFromPlate(plateCanvas, charBox) {
    const cropCanvas = document.createElement("canvas");
    const w = charBox.x2 - charBox.x1;
    const h = charBox.y2 - charBox.y1;
    const pad = Math.min(2, Math.floor(w * 0.1));
    const x1 = Math.max(0, charBox.x1 - pad);
    const y1 = Math.max(0, charBox.y1 - pad);
    const x2 = Math.min(plateCanvas.width, charBox.x2 + pad);
    const y2 = Math.min(plateCanvas.height, charBox.y2 + pad);
    cropCanvas.width = x2 - x1;
    cropCanvas.height = y2 - y1;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.drawImage(plateCanvas, x1, y1, cropCanvas.width, cropCanvas.height, 0, 0, cropCanvas.width, cropCanvas.height);
    return cropCanvas;
}

// ==================== Xử lý toàn bộ biển số ====================
async function processPlate(plateCanvas) {
    // Bước 1: Dùng CRAFT tìm các box ký tự
    let charBoxes = await detectCharacterBoxesWithCRAFT(plateCanvas);
    if (!charBoxes.length) {
        log("Không tìm thấy ký tự nào bằng CRAFT", true);
        return "";
    }
    // Sắp xếp: theo dòng (y) rồi theo x
    charBoxes.sort((a,b) => {
        if (Math.abs(a.y1 - b.y1) < 10) return a.x1 - b.x1;
        return a.y1 - b.y1;
    });
    // Nhận diện từng ký tự
    let plateText = "";
    for (let box of charBoxes) {
        const charCanvas = cropCharFromPlate(plateCanvas, box);
        const ch = await recognizeCharFromCanvas(charCanvas);
        if (ch) plateText += ch;
        else plateText += "?";
    }
    // Clean và format
    plateText = plateText.replace(/\?/g, '');
    if (plateText.length >= 3) {
        plateText = plateText.slice(0,3) + "-" + plateText.slice(3);
    }
    return plateText;
}

// ==================== Vẽ kết quả ====================
function drawResults(plateBoxes, charBoxesOnPlate, recognizedText, plateCanvas) {
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

// ==================== Luồng chính detect ====================
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
            // Crop biển số từ video
            const pad = CHAR_PADDING_RATIO;
            const cropX1 = Math.max(0, bestPlate.x1 - (bestPlate.x2-bestPlate.x1)*pad);
            const cropY1 = Math.max(0, bestPlate.y1 - (bestPlate.y2-bestPlate.y1)*pad);
            const cropX2 = Math.min(video.videoWidth, bestPlate.x2 + (bestPlate.x2-bestPlate.x1)*pad);
            const cropY2 = Math.min(video.videoHeight, bestPlate.y2 + (bestPlate.y2-bestPlate.y1)*pad);
            const plateCanvas = document.createElement("canvas");
            plateCanvas.width = cropX2 - cropX1;
            plateCanvas.height = cropY2 - cropY1;
            const pctx = plateCanvas.getContext("2d");
            pctx.drawImage(video, cropX1, cropY1, plateCanvas.width, plateCanvas.height, 0, 0, plateCanvas.width, plateCanvas.height);
            
            recognizedPlateText = await processPlate(plateCanvas);
            if (recognizedPlateText) finalText = `🔢 ${recognizedPlateText}`;
            else finalText = `⚠️ Biển số (${Math.round(bestPlate.score*100)}%) nhưng không đọc được ký tự`;
        }
        resultTextDiv.innerText = finalText;
        drawResults(plateBoxes, null, recognizedPlateText, null);
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
        log("⏳ Tải model CRAFT...");
        sessionCraft = await loadModel(CRAFT_MODEL_PATH);
        log("⏳ Tải model CRNN...");
        sessionRecognizer = await loadModel(RECOGNIZER_PATH);
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
