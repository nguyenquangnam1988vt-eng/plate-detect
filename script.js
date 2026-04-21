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
const CHAR_CNN_MODEL_PATH = "/model/char_cnn.onnx";   // đã chuyển từ weight.h5
const INPUT_SIZE = 640;

// Ngưỡng cho biển số
const CONF_THRESHOLD_PLATE = 0.5;
const IOU_THRESHOLD_PLATE = 0.45;
const MIN_BOX_SIZE_PLATE = 30;
const MIN_ASPECT_RATIO = 1.5;
const MAX_ASPECT_RATIO = 5.0;

// FPS giới hạn (2 khung/giây)
const FPS_LIMIT = 2;

// Bảng ánh xạ class (giống ALPHA_DICT, class 31 là background)
const CHAR_CLASSES = [
    'A','B','C','D','E','F','G','H','K','L','M','N','P',
    'R','S','T','U','V','X','Y','Z','0','1','2','3','4',
    '5','6','7','8','9','Background'
];

// Ngưỡng cho segmentation (theo code Python)
const CHAR_ASPECT_MIN = 0.1;
const CHAR_ASPECT_MAX = 1.0;
const CHAR_SOLIDITY_MIN = 0.1;
const CHAR_HEIGHT_RATIO_MIN = 0.35;
const CHAR_HEIGHT_RATIO_MAX = 2.0;

// ==================== Biến toàn cục ====================
let sessionPlate = null;
let sessionChar = null;      // model CNN cho ký tự
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

// ==================== Tiền xử lý ảnh (letterbox) cho YOLO ====================
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

// ==================== Parse YOLO output (cho biển số) ====================
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

// ==================== Crop ảnh biển số từ video ====================
function cropImageFromVideo(box, paddingRatio = 0.2) {
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

// ==================== Chuyển đổi ảnh ký tự thành vuông (convert2Square) ====================
function convertToSquare(imgMat) {
    // imgMat là cv.Mat (grayscale, binary)
    const size = Math.max(imgMat.rows, imgMat.cols);
    const square = new cv.Mat.zeros(size, size, cv.CV_8UC1);
    const offsetX = (size - imgMat.cols) / 2;
    const offsetY = (size - imgMat.rows) / 2;
    const roi = square.roi(new cv.Rect(offsetX, offsetY, imgMat.cols, imgMat.rows));
    imgMat.copyTo(roi);
    roi.delete();
    return square;
}

// ==================== Nhận diện ký tự bằng CNN (segmentation + classification) ====================
async function detectCharactersOnCrop(plateCanvas) {
    if (!cvReady) {
        throw new Error("OpenCV chưa sẵn sàng");
    }
    if (!sessionChar) {
        log("Đang tải model CNN cho ký tự...");
        sessionChar = await loadModel(CHAR_CNN_MODEL_PATH);
    }

    // Chuyển canvas plate thành cv.Mat
    let src = cv.imread(plateCanvas);
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    
    // Bước 1: Lấy kênh V từ HSV (tương tự Python)
    let hsv = new cv.Mat();
    cv.cvtColor(src, hsv, cv.COLOR_RGBA2HSV);
    let channels = new cv.MatVector();
    cv.split(hsv, channels);
    let V = channels.get(2);  // kênh Value
    
    // Bước 2: Adaptive threshold (giả lập threshold_local)
    // OpenCV.js không có threshold_local trực tiếp, dùng adaptiveThreshold thay thế
    let thresh = new cv.Mat();
    cv.adaptiveThreshold(V, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 10);
    // Đảo ngược (vì trong Python họ dùng bitwise_not)
    cv.bitwise_not(thresh, thresh);
    
    // Resize về chiều rộng 400
    let resized = new cv.Mat();
    const targetWidth = 400;
    const scale = targetWidth / thresh.cols;
    const targetHeight = Math.round(thresh.rows * scale);
    cv.resize(thresh, resized, new cv.Size(targetWidth, targetHeight), 0, 0, cv.INTER_LINEAR);
    
    // Median blur
    let blurred = new cv.Mat();
    cv.medianBlur(resized, blurred, 5);
    
    // Connected components (phân tích thành phần liên thông)
    let labels = new cv.Mat();
    let stats = new cv.Mat();
    let centroids = new cv.Mat();
    let numLabels = cv.connectedComponentsWithStats(blurred, labels, stats, centroids, 8, cv.CV_32S);
    
    const candidates = [];  // lưu { mat, y, x }
    
    for (let label = 1; label < numLabels; label++) {
        // Tạo mask cho label hiện tại
        let mask = new cv.Mat.zeros(blurred.rows, blurred.cols, cv.CV_8UC1);
        for (let i = 0; i < blurred.rows; i++) {
            for (let j = 0; j < blurred.cols; j++) {
                if (labels.intAt(i, j) === label) {
                    mask.ucharPtr(i, j)[0] = 255;
                }
            }
        }
        
        // Tìm contour
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        
        if (contours.size() > 0) {
            // Lấy contour lớn nhất
            let maxArea = 0;
            let bestContour = null;
            for (let i = 0; i < contours.size(); i++) {
                let cnt = contours.get(i);
                let area = cv.contourArea(cnt);
                if (area > maxArea) {
                    maxArea = area;
                    bestContour = cnt;
                }
            }
            if (bestContour) {
                let rect = cv.boundingRect(bestContour);
                let x = rect.x;
                let y = rect.y;
                let w = rect.width;
                let h = rect.height;
                
                let aspectRatio = w / h;
                let solidity = maxArea / (w * h);
                let heightRatio = h / blurred.rows;  // tỷ lệ so với chiều cao vùng ảnh đã resize
                
                if (aspectRatio > CHAR_ASPECT_MIN && aspectRatio < CHAR_ASPECT_MAX &&
                    solidity > CHAR_SOLIDITY_MIN &&
                    heightRatio > CHAR_HEIGHT_RATIO_MIN && heightRatio < CHAR_HEIGHT_RATIO_MAX) {
                    // Cắt ký tự
                    let charMat = mask.roi(new cv.Rect(x, y, w, h));
                    let squareMat = convertToSquare(charMat);
                    let resizedChar = new cv.Mat();
                    cv.resize(squareMat, resizedChar, new cv.Size(28, 28), 0, 0, cv.INTER_AREA);
                    // Lưu lại (y là tọa độ dùng để phân dòng)
                    candidates.push({ mat: resizedChar, y: y, x: x });
                    charMat.delete();
                    squareMat.delete();
                }
            }
        }
        mask.delete();
        contours.delete();
        hierarchy.delete();
    }
    
    // Dự đoán từng ký tự
    const charImages = [];
    const coords = [];
    for (let cand of candidates) {
        // Chuyển mat thành tensor float32 [1,1,28,28] (batch, channel, height, width)
        const data = new Float32Array(28 * 28);
        for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
                data[i * 28 + j] = cand.mat.ucharPtr(i, j)[0] / 255.0;
            }
        }
        const tensor = new ort.Tensor("float32", data, [1, 1, 28, 28]);
        charImages.push(tensor);
        coords.push({ y: cand.y, x: cand.x });
        cand.mat.delete();
    }
    
    if (charImages.length === 0) {
        src.delete(); gray.delete(); hsv.delete(); channels.delete(); V.delete();
        thresh.delete(); resized.delete(); blurred.delete(); labels.delete(); stats.delete(); centroids.delete();
        return { boxes: [], text: "" };
    }
    
    // Chạy batch inference (có thể chạy tuần tự nếu model không hỗ trợ batch)
    const results = [];
    for (let i = 0; i < charImages.length; i++) {
        const feed = { [sessionChar.inputNames[0]]: charImages[i] };
        const output = await sessionChar.run(feed);
        const outputTensor = output[sessionChar.outputNames[0]];
        const probs = outputTensor.data;
        let maxIdx = 0;
        let maxProb = probs[0];
        for (let j = 1; j < probs.length; j++) {
            if (probs[j] > maxProb) {
                maxProb = probs[j];
                maxIdx = j;
            }
        }
        if (maxIdx !== 31) {  // bỏ qua background
            results.push({ label: CHAR_CLASSES[maxIdx], score: maxProb, coord: coords[i] });
        }
    }
    
    // Phân dòng (giống format trong Python)
    if (results.length === 0) {
        src.delete(); gray.delete(); hsv.delete(); channels.delete(); V.delete();
        thresh.delete(); resized.delete(); blurred.delete(); labels.delete(); stats.delete(); centroids.delete();
        return { boxes: [], text: "" };
    }
    
    // Sắp xếp theo y (dòng)
    results.sort((a,b) => a.coord.y - b.coord.y);
    const firstY = results[0].coord.y;
    const firstLine = [];
    const secondLine = [];
    for (let r of results) {
        if (Math.abs(r.coord.y - firstY) < 20) {
            firstLine.push(r);
        } else {
            secondLine.push(r);
        }
    }
    // Sắp xếp theo x
    firstLine.sort((a,b) => a.coord.x - b.coord.x);
    secondLine.sort((a,b) => a.coord.x - b.coord.x);
    
    let plateText = firstLine.map(r => r.label).join('');
    if (secondLine.length > 0) {
        plateText += '-' + secondLine.map(r => r.label).join('');
    }
    
    // Tạo bounding boxes cho kết quả (tạm thời không có tọa độ thực trên canvas gốc, chỉ trả về text)
    const boxes = results.map(r => ({
        x1: 0, y1: 0, x2: 0, y2: 0, label: r.label, score: r.score
    }));
    
    // Dọn dẹp
    src.delete(); gray.delete(); hsv.delete(); channels.delete(); V.delete();
    thresh.delete(); resized.delete(); blurred.delete(); labels.delete(); stats.delete(); centroids.delete();
    
    return { boxes: boxes, text: plateText };
}

// ==================== Vẽ kết quả ====================
function drawResults(plateBoxes, charResults) {
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
    // Vẽ kết quả text lên góc
    if (charResults && charResults.text) {
        ctx.fillStyle = "#ffaa44";
        ctx.font = "bold 20px monospace";
        ctx.fillText(charResults.text, 10, 40);
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
            const cropCanvas = cropImageFromVideo(bestPlate, 0.2);
            if (cropCanvas) {
                try {
                    charData = await detectCharactersOnCrop(cropCanvas);
                    if (charData.text.length > 0) finalText = `🔢 ${charData.text}`;
                    else finalText = `⚠️ Biển số (${Math.round(bestPlate.score*100)}%) nhưng không đọc được ký tự`;
                } catch (err) {
                    log("Lỗi nhận diện ký tự: " + err.message, true);
                    finalText = `⚠️ Lỗi đọc ký tự`;
                }
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
        log("⏳ Đang tải model biển số...");
        sessionPlate = await loadModel(PLATE_MODEL_PATH);
        log("⏳ Đang tải model CNN ký tự (đã chuyển từ weight.h5)...");
        // sessionChar sẽ được load khi cần trong detectCharactersOnCrop
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
