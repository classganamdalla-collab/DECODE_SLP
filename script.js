import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// ── DOM ──────────────────────────────────────────────
const video             = document.getElementById("webcam");
const canvas            = document.getElementById("outputCanvas");
const canvasCtx         = canvas.getContext("2d");
const startBtn          = document.getElementById("startBtn");
const addWordBtn        = document.getElementById("addWordBtn");
const clearSentenceBtn  = document.getElementById("clearSentenceBtn");
const speakSentenceBtn  = document.getElementById("speakSentenceBtn");
const listenBtn         = document.getElementById("listenBtn");
const speechPanel       = document.getElementById("speechPanel");
const speechFinal       = document.getElementById("speechFinal");
const speechInterim     = document.getElementById("speechInterim");
const clearSpeechBtn    = document.getElementById("clearSpeechBtn");
const datasetStatus     = document.getElementById("datasetStatus");
const cameraStatus      = document.getElementById("cameraStatus");
const handStatus        = document.getElementById("handStatus");
const runStatus         = document.getElementById("runStatus");
const currentPrediction = document.getElementById("currentPrediction");
const predictionScore   = document.getElementById("predictionScore");
const holdStatus        = document.getElementById("holdStatus");
const holdBarFill       = document.getElementById("holdBarFill");
const sentenceOutput    = document.getElementById("sentenceOutput");
const labelChips        = document.getElementById("labelChips");

// ── 설정 ─────────────────────────────────────────────
let   LIVE_BUFFER_MAX    = 100; // metadata.json에서 덮어씀
const ML_FEATURE_DIM     = 90;
const MIN_SCORE_SHOW     = 0.70;
const MIN_SCORE_LOCK     = 0.75;
const MIN_MARGIN         = 0.20;
const MIN_GESTURE_FRAMES = 15;   // 최소 녹화 프레임 (너무 짧으면 무시)
const NO_HAND_END_FRAMES = 8;    // 손이 N프레임 없으면 제스처 종료로 판단
const COOLDOWN_MS        = 1500; // 인식 후 다음 제스처까지 대기 시간
const LOCK_HOLD_MS       = 700;  // 신뢰도 유지 시간 (ms)
const FACE_KEY_POINTS    = [4, 10, 13, 33, 152, 234, 263, 454];

// ── 상태 ─────────────────────────────────────────────
let handLandmarker      = null;
let faceLandmarker      = null;
let drawingUtils        = null;
let mlModel             = null;
let webcamRunning       = false;
let lastVideoTime       = -1;
let appStarted          = false;
let autoAddEnabled      = true;

let labelsConfig        = [];
let targetLabels        = [];
let sentenceWords       = [];
let latestDetectedHands = [];

// 제스처 구간 감지 상태
let gestureState    = 'waiting';
let gestureBuffer   = [];
let noHandCount     = 0;
let lastResult      = null;
let liveInferCount  = 0;  // 실시간 추론 프레임 카운터
let lockStartTime   = 0;  // 신뢰도 유지 시작 시각 (ms)
let lastFaceResults = null;
let lastHandResults = null;
let faceFrameCount  = 0;
let handFrameCount  = 0;

// MediaPipe 추론 전용 640×480 캔버스 (표시용 비디오 해상도와 분리)
const inferCanvas = document.createElement('canvas');
inferCanvas.width  = 640;
inferCanvas.height = 480;
const inferCtx = inferCanvas.getContext('2d');

// ── 1. labels.json 로드 ───────────────────────────────
async function loadLabels() {
  try {
    const res = await fetch("./labels.json");
    if (!res.ok) throw new Error();
    const json = await res.json();
    labelsConfig = json.labels || [];
    targetLabels = labelsConfig.map(l => l.id);
    renderLabelChips();
  } catch {
    labelsConfig = [];
    targetLabels = [];
  }
}

function renderLabelChips() {
  if (!labelChips) return;
  labelChips.innerHTML = labelsConfig
    .filter(l => l.id !== "기타")
    .map(l => `<span class="chip">${l.korean}</span>`)
    .join("");
}

// ── 2. ML 모델 로드 ───────────────────────────────────
async function loadMLModel() {
  try {
    datasetStatus.textContent = "ML 모델 로딩 중...";
    await tf.setBackend('webgl');
    await tf.ready();
    const meta = await fetch("./model/metadata.json").then(r => r.json()).catch(() => null);
    if (meta && meta.max_sequence_length) LIVE_BUFFER_MAX = meta.max_sequence_length;
    mlModel = await tf.loadLayersModel("./model/tfjs_model/model.json");
    datasetStatus.textContent = "✅ ML 모델 로딩 완료";
  } catch (e) {
    datasetStatus.textContent = "❌ " + e.message;
    console.error(e);
  }
}

// ── 3. MediaPipe 초기화 ───────────────────────────────
async function createLandmarkers() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    },
    runningMode: "VIDEO",
    numFaces: 1
  });
  drawingUtils = new DrawingUtils(canvasCtx);
}

// ── 4. 카메라 ─────────────────────────────────────────
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => { video.onloadedmetadata = () => resolve(); });
    await video.play();
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    webcamRunning = true;
    cameraStatus.textContent = "카메라 연결 완료";
  } catch (e) {
    cameraStatus.textContent = "카메라 연결 실패";
    console.error(e);
  }
}

// ── 5. 랜드마크 그리기 ────────────────────────────────
function drawResults(handResults, faceResults) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  canvasCtx.translate(canvas.width, 0);
  canvasCtx.scale(-1, 1);

  if (handResults && handResults.landmarks) {
    for (const lm of handResults.landmarks) {
      drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS,
        { lineWidth: 2, color: "rgba(95,212,196,0.6)" });
      drawingUtils.drawLandmarks(lm, { radius: 3, color: "rgba(79,182,245,0.75)" });
    }
  }

  if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
    const faceLM = faceResults.faceLandmarks[0];
    for (const idx of FACE_KEY_POINTS) {
      const pt = faceLM[idx];
      if (!pt) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(pt.x * canvas.width, pt.y * canvas.height, 5, 0, Math.PI * 2);
      canvasCtx.fillStyle = "rgba(79,182,245,0.5)";
      canvasCtx.fill();
    }
  }

  canvasCtx.restore();
}

// ── 6. 메인 루프 ──────────────────────────────────────
function predictWebcam() {
  if (!appStarted || !webcamRunning || !handLandmarker) return;
  const now = performance.now();

  // 그리기는 항상 (캐시된 결과 사용) → 점이 끊기지 않음
  drawResults(lastHandResults, lastFaceResults);

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // 추론 캔버스에 현재 프레임 축소 복사
    inferCtx.drawImage(video, 0, 0, 640, 480);

    // 손 감지: 3프레임마다
    handFrameCount++;
    if (handFrameCount % 3 === 0) {
      lastHandResults = handLandmarker.detectForVideo(inferCanvas, now);
      const prev = latestDetectedHands.length;
      latestDetectedHands = cloneHands(lastHandResults);
      const cur = latestDetectedHands.length;
      if (prev !== cur) {
        handStatus.textContent = cur > 0 ? `손 감지됨: ${cur}개` : "손 감지되지 않음";
      }
    }

    // 얼굴 감지: 9프레임마다
    faceFrameCount++;
    if (faceFrameCount % 9 === 0) {
      lastFaceResults = faceLandmarker.detectForVideo(inferCanvas, now);
    }

    updateGestureBuffer(lastHandResults, lastFaceResults);
  }

  requestAnimationFrame(predictWebcam);
}

// ── 7. 손 데이터 복제 ─────────────────────────────────
function cloneHands(handResults) {
  if (!handResults || !handResults.landmarks) return [];
  return handResults.landmarks.map((lm, i) => ({
    handedness: handResults.handedness?.[i]?.[0]?.categoryName || "Unknown",
    landmarks: lm.map(p => ({ x: p.x, y: p.y, z: p.z }))
  }));
}

// ── 8. 특징 추출 (상대 좌표 + 손바닥 법선 벡터) ──────
function extractMLFrame(handResults, faceResults) {
  const features = [];

  // 손: 손목(landmark 0) 기준 상대 좌표 × 21개 = 63
  if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
    const lm    = handResults.landmarks[0];
    const wrist = lm[0];
    for (const pt of lm) {
      features.push(pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z);
    }

    // 손바닥 법선 벡터 (palm normal): cross(wrist→검지MCP, wrist→소지MCP) 정규화 = 3
    // 손바닥이 향하는 방향을 직접 인코딩해서 '나'/'너' 구분력 향상
    const idxMcp  = lm[5];
    const pinkMcp = lm[17];
    const v1x = idxMcp.x  - wrist.x,  v1y = idxMcp.y  - wrist.y,  v1z = idxMcp.z  - wrist.z;
    const v2x = pinkMcp.x - wrist.x,  v2y = pinkMcp.y - wrist.y,  v2z = pinkMcp.z - wrist.z;
    const nx = v1y * v2z - v1z * v2y;
    const ny = v1z * v2x - v1x * v2z;
    const nz = v1x * v2y - v1y * v2x;
    const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nLen > 1e-6) features.push(nx / nLen, ny / nLen, nz / nLen);
    else             features.push(0, 0, 0);
  } else {
    for (let i = 0; i < 66; i++) features.push(0);
  }

  // 얼굴: 코끝(첫 번째 키포인트) 기준 상대 좌표 × 8개 = 24
  if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
    const faceLM = faceResults.faceLandmarks[0];
    const nose   = faceLM[FACE_KEY_POINTS[0]];
    for (const idx of FACE_KEY_POINTS) {
      const pt = faceLM[idx];
      features.push(
        pt ? pt.x - nose.x : 0,
        pt ? pt.y - nose.y : 0,
        pt ? pt.z - nose.z : 0
      );
    }
  } else {
    for (let i = 0; i < 24; i++) features.push(0);
  }

  return features; // 63 + 3 + 24 = 90
}

// ── 9. 제스처 구간 감지 ───────────────────────────────
// 손 나타남 → 녹화 → 손 사라짐 → 분류 (학습 방식과 동일한 구조)
function updateGestureBuffer(handResults, faceResults) {
  if (gestureState === 'cooldown') return;

  const hasHand = latestDetectedHands.length > 0;

  // ── WAITING: 손 나타나면 녹화 시작
  if (gestureState === 'waiting') {
    if (hasHand) {
      gestureState   = 'recording';
      gestureBuffer  = [];
      noHandCount    = 0;
      liveInferCount = 0;
      lockStartTime  = 0;
      holdStatus.textContent        = "동작 중...";
      holdBarFill.style.width       = "0%";
      currentPrediction.textContent = "—";
      predictionScore.textContent   = "";
    }
    return;
  }

  // ── RECORDING: 프레임 수집
  if (gestureState === 'recording') {
    if (hasHand) {
      noHandCount = 0;
      gestureBuffer.push(extractMLFrame(handResults, faceResults));
      if (gestureBuffer.length > LIVE_BUFFER_MAX) gestureBuffer.shift();

      // 5프레임마다 추론 (매 프레임 추론하면 너무 느림)
      liveInferCount++;
      if (liveInferCount % 8 === 0 && gestureBuffer.length >= MIN_GESTURE_FRAMES) {
        const live = runMLInference(gestureBuffer);
        const t = performance.now();
        if (live && live.score >= MIN_SCORE_SHOW && live.label !== "기타") {
          const found = labelsConfig.find(l => l.id === live.label);
          currentPrediction.textContent = found ? found.korean : live.label;
          predictionScore.textContent   = `신뢰도: ${(live.score * 100).toFixed(1)}%`;
          holdBarFill.style.width       = `${(live.score * 100).toFixed(0)}%`;

          if (live.score >= MIN_SCORE_LOCK) {
            if (lockStartTime === 0) lockStartTime = t;
            const elapsed = t - lockStartTime;
            const remaining = Math.max(0, LOCK_HOLD_MS - elapsed);
            holdStatus.textContent = remaining > 50
              ? `동작 유지... (${(remaining / 1000).toFixed(1)}s)`
              : "확정!";
            if (elapsed >= LOCK_HOLD_MS) {
              classifyGesture();
              return;
            }
          } else {
            lockStartTime = 0;
            holdStatus.textContent = "손을 내리면 확정돼요";
          }
        } else {
          lockStartTime = 0;
          currentPrediction.textContent = "...";
          predictionScore.textContent   = "";
          holdStatus.textContent        = "동작 중...";
          holdBarFill.style.width       = "0%";
        }
      }
    } else {
      noHandCount++;
      lockStartTime = 0;
      if (noHandCount >= NO_HAND_END_FRAMES) {
        if (gestureBuffer.length >= MIN_GESTURE_FRAMES) {
          classifyGesture();
        } else {
          resetGesture();
          holdStatus.textContent = "너무 짧습니다. 다시 해주세요";
        }
      }
    }
  }
}

// ── 10. 제스처 분류 ───────────────────────────────────
function classifyGesture() {
  const best = runMLInference(gestureBuffer);

  gestureState  = 'cooldown';
  gestureBuffer = [];

  if (!best || best.score < MIN_SCORE_SHOW || best.label === "기타") {
    currentPrediction.textContent = "...";
    predictionScore.textContent   = "";
    holdStatus.textContent        = "인식 실패 — 다시 해주세요";
    holdBarFill.style.width       = "0%";
    lastResult = null;
    setTimeout(resetGesture, COOLDOWN_MS);
    return;
  }

  const found = labelsConfig.find(l => l.id === best.label);
  const word  = found ? found.korean : best.label;

  currentPrediction.textContent = word;
  predictionScore.textContent   = `신뢰도: ${(best.score * 100).toFixed(1)}%`;
  holdBarFill.style.width       = "100%";
  lastResult = best;

  if (autoAddEnabled && best.score >= MIN_SCORE_LOCK) {
    addWordToSentence(best.label);
  } else if (best.score >= MIN_SCORE_LOCK) {
    holdStatus.textContent = `✅ 추가 가능: ${word} (버튼 누르세요)`;
  } else {
    holdStatus.textContent = `신뢰도 부족 (${(best.score * 100).toFixed(1)}%) — 다시 해주세요`;
  }

  setTimeout(resetGesture, COOLDOWN_MS);
}

// ── 11. ML 추론 ───────────────────────────────────────
function runMLInference(frames) {
  if (!mlModel || frames.length < MIN_GESTURE_FRAMES) return null;

  const data = new Float32Array(LIVE_BUFFER_MAX * ML_FEATURE_DIM);
  const len  = Math.min(frames.length, LIVE_BUFFER_MAX);
  for (let i = 0; i < len; i++) {
    for (let j = 0; j < ML_FEATURE_DIM; j++) {
      data[i * ML_FEATURE_DIM + j] = frames[i][j] || 0;
    }
  }

  const inputTensor  = tf.tensor3d(data, [1, LIVE_BUFFER_MAX, ML_FEATURE_DIM]);
  const outputTensor = mlModel.predict(inputTensor);
  const probs        = outputTensor.dataSync();
  inputTensor.dispose();
  outputTensor.dispose();

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  let secondScore = 0;
  for (let i = 0; i < probs.length; i++) {
    if (i !== bestIdx && probs[i] > secondScore) secondScore = probs[i];
  }
  if (probs[bestIdx] - secondScore < MIN_MARGIN) return null;

  return { label: targetLabels[bestIdx], score: probs[bestIdx] };
}

// ── 12. 단어 추가 ─────────────────────────────────────
function addWordToSentence(labelId) {
  const found = labelsConfig.find(l => l.id === labelId);
  const word  = found ? found.korean : labelId;
  sentenceWords.push(word);
  updateSentenceUI();
  scheduleAutoSpeak();
  holdStatus.textContent  = `✅ "${word}" 추가됨!`;
}

function resetGesture() {
  gestureState  = 'waiting';
  gestureBuffer = [];
  noHandCount   = 0;
  holdBarFill.style.width = "0%";
  if (appStarted) holdStatus.textContent = "손을 카메라에 보여주세요";
}

// ── 13. 수동 추가 (버튼) ──────────────────────────────
function addWord() {
  if (!lastResult || lastResult.label === "기타") {
    alert("아직 인식된 단어가 없습니다.");
    return;
  }
  addWordToSentence(lastResult.label);
  lastResult = null;
}

function clearSentence() { clearTimeout(autoSpeakTimer); sentenceWords = []; updateSentenceUI(); }

function updateSentenceUI() {
  if (sentenceWords.length === 0) {
    sentenceOutput.innerHTML = '<span class="sentence-empty">아직 추가된 단어가 없습니다.</span>';
    return;
  }
  sentenceOutput.innerHTML = sentenceWords
    .map((w, i) => `<span class="sentence-chip">${w}<button class="sentence-chip-del" data-idx="${i}">×</button></span>`)
    .join("");
  sentenceOutput.querySelectorAll(".sentence-chip-del").forEach(btn => {
    btn.addEventListener("click", e => {
      sentenceWords.splice(parseInt(e.currentTarget.dataset.idx), 1);
      updateSentenceUI();
    });
  });
}

// 자연스러운 한국어 어순으로 재배열 후 TTS
const WORD_PRIORITY = {
  "안녕하세요": 0,
  "아이스": 1, "뜨거운": 1,
  "아메리카노": 2,
  "제일": 3, "큰 걸로": 3,
  "2잔": 4,
  "테이크아웃": 5, "해주세요": 6, "주세요": 6,
  "와이파이": 7, "있나요?": 8,
  "포인트": 9, "카드": 10, "영수증": 11,
  "감사합니다": 12,
};

// 단어 하나일 때 자연스러운 완성 문장
const SINGLE_WORD_TTS = {
  "안녕하세요":  "안녕하세요",
  "아메리카노":  "아메리카노 주세요",
  "뜨거운":      "뜨거운 걸로 주세요",
  "아이스":      "아이스로 주세요",
  "2잔":         "두 잔 주세요",
  "제일":        "제일 큰 걸로 주세요",
  "큰 걸로":     "큰 걸로 주세요",
  "테이크아웃":  "테이크아웃으로 해주세요",
  "포인트":      "포인트 할게요",
  "카드":        "카드로 결제할게요",
  "영수증":      "영수증 주세요",
  "와이파이":    "와이파이 있나요?",
  "감사합니다":  "감사합니다",
};

const TERMINAL_SET = new Set(["주세요", "해주세요", "있나요?", "감사합니다", "안녕하세요"]);

function getWordTTS(korean) {
  const found = labelsConfig.find(l => l.korean === korean);
  return found ? found.tts : korean;
}

function buildNaturalTTS(sortedKorean) {
  if (sortedKorean.length === 1) {
    return SINGLE_WORD_TTS[sortedKorean[0]] ?? getWordTTS(sortedKorean[0]);
  }
  if (sortedKorean.some(w => TERMINAL_SET.has(w))) {
    return sortedKorean.map(getWordTTS).join(" ");
  }
  // 결제 키워드 감지 → 자연스러운 마무리 추가
  if (sortedKorean.includes("카드")) {
    const rest = sortedKorean.filter(w => w !== "카드").map(getWordTTS);
    return rest.length ? `${rest.join(" ")} 카드로 결제할게요` : "카드로 결제할게요";
  }
  if (sortedKorean.includes("포인트")) {
    const rest = sortedKorean.filter(w => w !== "포인트").map(getWordTTS);
    return rest.length ? `${rest.join(" ")} 포인트 할게요` : "포인트 할게요";
  }
  if (sortedKorean.includes("영수증")) {
    const rest = sortedKorean.filter(w => w !== "영수증").map(getWordTTS);
    return rest.length ? `${rest.join(" ")} 영수증 주세요` : "영수증 주세요";
  }
  return sortedKorean.map(getWordTTS).join(" ") + " 주세요";
}

// 자동 읽기 (단어 추가 후 N초 경과 시 자동 발화)
let autoSpeakEnabled = true;
let autoSpeakTimer   = null;
const AUTO_SPEAK_DELAY = 3000;

function scheduleAutoSpeak() {
  if (!autoSpeakEnabled || sentenceWords.length === 0) return;
  clearTimeout(autoSpeakTimer);
  autoSpeakTimer = setTimeout(() => {
    if (sentenceWords.length > 0) speakSentence();
  }, AUTO_SPEAK_DELAY);
}

function speakSentence() {
  if (sentenceWords.length === 0) { alert("읽을 문장이 없습니다."); return; }
  clearTimeout(autoSpeakTimer);
  const sorted = [...sentenceWords].sort((a, b) =>
    (WORD_PRIORITY[a] ?? 99) - (WORD_PRIORITY[b] ?? 99)
  );
  const text = buildNaturalTTS(sorted);
  const utt  = new SpeechSynthesisUtterance(text);
  utt.lang   = "ko-KR";
  utt.onend  = () => clearSentence();
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utt);
}

// ── 이벤트 ───────────────────────────────────────────
const statusDot = document.getElementById("statusDot");
startBtn.addEventListener("click", () => {
  appStarted = !appStarted;
  resetGesture();
  lastResult = null;
  if (appStarted) {
    startBtn.textContent = "■";
    startBtn.classList.add("recording");
    statusDot.classList.add("active");
    runStatus.textContent         = "인식 중";
    currentPrediction.textContent = "—";
    predictionScore.textContent   = "";
    predictWebcam();
  } else {
    startBtn.textContent = "▶";
    startBtn.classList.remove("recording");
    statusDot.classList.remove("active");
    runStatus.textContent         = "중지됨";
    currentPrediction.textContent = "—";
    predictionScore.textContent   = "";
    holdStatus.textContent        = "안정화 대기 중";
    holdBarFill.style.width       = "0%";
  }
});

addWordBtn.addEventListener("click", addWord);
addWordBtn.addEventListener("dblclick", () => {
  autoAddEnabled = !autoAddEnabled;
  addWordBtn.classList.toggle("manual-mode", !autoAddEnabled);
  if (appStarted) runStatus.textContent = autoAddEnabled ? "인식 중" : "인식 중 (수동)";
});

clearSentenceBtn.addEventListener("click", clearSentence);
speakSentenceBtn.addEventListener("click", speakSentence);
speakSentenceBtn.addEventListener("dblclick", () => {
  autoSpeakEnabled = !autoSpeakEnabled;
  speakSentenceBtn.classList.toggle("auto-mode", autoSpeakEnabled);
  speakSentenceBtn.textContent = autoSpeakEnabled ? "🔊 자동" : "🔊 읽기";
  if (!autoSpeakEnabled) clearTimeout(autoSpeakTimer);
});

// ── STT (상대방 음성 → 텍스트) ───────────────────────
let recognition  = null;
let isListening  = false;

function initSTT() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    listenBtn.textContent = "🎤 미지원";
    listenBtn.disabled    = true;
    return;
  }
  recognition = new SR();
  recognition.lang            = "ko-KR";
  recognition.continuous      = true;
  recognition.interimResults  = true;

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        speechFinal.textContent = t.trim();
        speechInterim.textContent = "";
      } else {
        interim += t;
      }
    }
    if (interim) speechInterim.textContent = interim;
    speechPanel.classList.toggle("has-content", speechFinal.textContent.trim() !== "");
  };

  recognition.onend = () => { if (isListening) recognition.start(); };

  recognition.onerror = (e) => {
    if (e.error !== "no-speech") speechInterim.textContent = `오류: ${e.error}`;
  };
}

function toggleListening() {
  if (!recognition) return;
  isListening = !isListening;
  if (isListening) {
    speechFinal.textContent   = "";
    speechInterim.textContent = "";
    speechPanel.classList.remove("hidden", "has-content");
    listenBtn.textContent = "🔴 듣기 중지";
    listenBtn.classList.add("listen-active");
    recognition.start();
  } else {
    recognition.stop();
    speechPanel.classList.add("hidden");
    speechPanel.classList.remove("has-content");
    listenBtn.textContent = "🎤 듣기";
    listenBtn.classList.remove("listen-active");
  }
}

listenBtn.addEventListener("click", toggleListening);
clearSpeechBtn.addEventListener("click", () => {
  speechFinal.textContent   = "";
  speechInterim.textContent = "";
  speechPanel.classList.remove("has-content");
});

// ── 앱 시작 ───────────────────────────────────────────
const launchBtn = document.getElementById("launchBtn");
const brandBtn  = document.getElementById("brandBtn");
let landmarkersReady = false;
let sttReady         = false;

async function preload() {
  await loadLabels();
  await loadMLModel();
  launchBtn.textContent = "시작하기";
  launchBtn.disabled    = false;
}

async function launchApp() {
  if (!landmarkersReady) {
    cameraStatus.textContent = "MediaPipe 준비 중...";
    await createLandmarkers();
    landmarkersReady = true;
  }
  cameraStatus.textContent = "카메라 연결 중...";
  await setupCamera();
  runStatus.textContent = "대기 중";
  if (!sttReady) { initSTT(); sttReady = true; }
}

launchBtn.addEventListener("click", async () => {
  document.getElementById("startScreen").classList.add("hidden");
  document.getElementById("mainApp").classList.remove("hidden");
  await launchApp();
});

brandBtn.addEventListener("click", () => {
  if (appStarted) {
    appStarted = false;
    startBtn.textContent = "▶";
    startBtn.classList.remove("recording");
    statusDot.classList.remove("active");
  }
  if (isListening) toggleListening();
  webcamRunning = false;
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  document.getElementById("mainApp").classList.add("hidden");
  document.getElementById("startScreen").classList.remove("hidden");
});

preload();
