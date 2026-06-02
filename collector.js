import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// ── DOM ──────────────────────────────────────────────
const video            = document.getElementById("webcam");
const canvas           = document.getElementById("outputCanvas");
const canvasCtx        = canvas.getContext("2d");
const labelGrid        = document.getElementById("labelGrid");
const situationBadge   = document.getElementById("situationBadge");
const sampleNote       = document.getElementById("sampleNote");
const startRecordBtn   = document.getElementById("startRecordBtn");
const stopRecordBtn    = document.getElementById("stopRecordBtn");
const exportJsonBtn    = document.getElementById("exportJsonBtn");
const mergeJsonBtn     = document.getElementById("mergeJsonBtn");
const mergeFileInput   = document.getElementById("mergeFileInput");
const clearSamplesBtn  = document.getElementById("clearSamplesBtn");
const cameraStatus     = document.getElementById("cameraStatus");
const handStatus       = document.getElementById("handStatus");
const recordStatus     = document.getElementById("recordStatus");
const frameCountStatus = document.getElementById("frameCountStatus");
const sampleCountStatus= document.getElementById("sampleCountStatus");
const samplesList      = document.getElementById("samplesList");
const labelSummary     = document.getElementById("labelSummary");
const hudLabel         = document.getElementById("hudLabel");
const hudRecord        = document.getElementById("hudRecord");

// ── 상태 ─────────────────────────────────────────────
let handLandmarker   = null;
let faceLandmarker   = null;
let drawingUtils     = null;
let webcamRunning    = false;
let lastVideoTime    = -1;
let latestHandCount  = 0;

let labelsConfig     = [];
let selectedLabel    = null;
let collectedSamples = [];
let autoRepeat       = false;

// 기록 상태
let isArmed          = false;   // 시작 버튼 누르고 손 감지를 기다리는 상태
let isRecording      = false;   // 실제 녹화 중
let currentRecording = null;

// 자동 종료 안정화
const NO_HAND_STOP_THRESHOLD = 3; // 손이 연속 3프레임 안 잡히면 종료
let noHandFrameStreak = 0;

// 얼굴 랜드마크 중 핵심 포인트만 저장
const FACE_KEY_POINTS = [4, 10, 13, 33, 152, 234, 263, 454];

// MediaPipe 추론 전용 640×480 캔버스
const inferCanvas = document.createElement('canvas');
inferCanvas.width  = 640;
inferCanvas.height = 480;
const inferCtx = inferCanvas.getContext('2d');

let lastHandResults = null;
let lastFaceResults = null;
let handFrameCount  = 0;
let faceFrameCount  = 0;

// ── 1. labels.json 로드 ───────────────────────────────
async function loadLabels() {
  try {
    const res = await fetch("./labels.json");
    if (!res.ok) throw new Error();
    const json = await res.json();
    labelsConfig = json.labels || [];
    situationBadge.textContent = json.situation || "기본";
    buildLabelGrid();
  } catch (e) {
    console.error(e);
    situationBadge.textContent = "labels.json 오류";
    labelsConfig = [
      { id: "나", korean: "나", tts: "나" },
      { id: "너", korean: "너", tts: "너" },
      { id: "기타", korean: "기타", tts: "기타" }
    ];
    buildLabelGrid();
  }
}

// ── 2. 라벨 버튼 생성 ────────────────────────────────
function buildLabelGrid() {
  labelGrid.innerHTML = "";
  for (const label of labelsConfig) {
    const btn = document.createElement("button");
    btn.className   = "label-btn";
    btn.textContent = label.korean;
    btn.dataset.id  = label.id;
    btn.addEventListener("click", () => selectLabel(label.id));
    labelGrid.appendChild(btn);
  }
  if (labelsConfig.length > 0) selectLabel(labelsConfig[0].id);
}

function selectLabel(id) {
  selectedLabel = id;
  document.querySelectorAll(".label-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.id === id);
  });
  const found = labelsConfig.find(l => l.id === id);
  hudLabel.textContent = found ? found.korean : id;
}

// ── 3. MediaPipe 초기화 (Hand + Face 동시) ───────────
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
      video: { width: 960, height: 540 },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => { video.onloadedmetadata = () => resolve(); });
    await video.play();
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    webcamRunning = true;
    cameraStatus.textContent = "📷 카메라 연결 완료";
  } catch (e) {
    console.error(e);
    cameraStatus.textContent = "📷 카메라 연결 실패";
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
        { lineWidth: 3, color: "#4f8cff" });
      drawingUtils.drawLandmarks(lm, { radius: 4, color: "#7c3aed" });
    }
  }

  if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
    const faceLM = faceResults.faceLandmarks[0];

    canvasCtx.restore();
    canvasCtx.save();
    canvasCtx.translate(canvas.width, 0);
    canvasCtx.scale(-1, 1);

    for (const idx of FACE_KEY_POINTS) {
      const pt = faceLM[idx];
      if (!pt) continue;
      const x = pt.x * canvas.width;
      const y = pt.y * canvas.height;
      canvasCtx.beginPath();
      canvasCtx.arc(x, y, 5, 0, Math.PI * 2);
      canvasCtx.fillStyle = "rgba(255, 200, 0, 0.85)";
      canvasCtx.fill();
    }
  }

  canvasCtx.restore();
}

// ── 6. 메인 루프 ──────────────────────────────────────
function predictWebcam() {
  if (!webcamRunning || !handLandmarker || !faceLandmarker) return;
  const now = performance.now();

  // 그리기는 항상 (캐시된 결과 사용) → 점이 끊기지 않음
  drawResults(lastHandResults, lastFaceResults);

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // 추론 캔버스에 현재 프레임 축소 복사
    inferCtx.drawImage(video, 0, 0, 640, 480);

    // 손 감지: 2프레임마다
    handFrameCount++;
    if (handFrameCount % 2 === 0) {
      lastHandResults = handLandmarker.detectForVideo(inferCanvas, now);
      const prev = latestHandCount;
      latestHandCount = lastHandResults.landmarks ? lastHandResults.landmarks.length : 0;
      if (prev !== latestHandCount) {
        handStatus.textContent = latestHandCount > 0
          ? `✋ 손 감지됨: ${latestHandCount}개`
          : "✋ 손 감지되지 않음";
      }
    }

    // 얼굴 감지: 6프레임마다
    faceFrameCount++;
    if (faceFrameCount % 6 === 0) {
      lastFaceResults = faceLandmarker.detectForVideo(inferCanvas, now);
    }

    handleAutoRecordingFlow(lastHandResults, lastFaceResults);
  }

  requestAnimationFrame(predictWebcam);
}

// ── 7. 얼굴 핵심 포인트 추출 ─────────────────────────
function extractFaceKeyPoints(faceResults) {
  if (!faceResults || !faceResults.faceLandmarks || faceResults.faceLandmarks.length === 0) {
    return FACE_KEY_POINTS.map(() => ({ x: 0, y: 0, z: 0 }));
  }
  const faceLM = faceResults.faceLandmarks[0];
  return FACE_KEY_POINTS.map(idx => {
    const pt = faceLM[idx];
    return pt ? { x: pt.x, y: pt.y, z: pt.z } : { x: 0, y: 0, z: 0 };
  });
}

// ── 8. 손 데이터 복제 ─────────────────────────────────
function cloneHands(handResults) {
  if (!handResults || !handResults.landmarks) return [];
  return handResults.landmarks.map((lm, i) => ({
    handedness: handResults.handedness?.[i]?.[0]?.categoryName || "Unknown",
    landmarks: lm.map(p => ({ x: p.x, y: p.y, z: p.z }))
  }));
}

// ── 9. 프레임 기록 ────────────────────────────────────
function addFrameToRecording(handResults, faceResults) {
  if (!isRecording || !currentRecording) return;

  const hands = cloneHands(handResults);
  if (hands.length === 0) return;

  const faceKeyPoints = extractFaceKeyPoints(faceResults);

  currentRecording.frames.push({
    timestamp: Date.now(),
    hands,
    face: faceKeyPoints
  });

  frameCountStatus.textContent = `프레임: ${currentRecording.frames.length}`;
}

// ── 10. 시작 버튼 → 대기 상태 진입 ───────────────────
function armRecording() {
  if (isArmed || isRecording) {
    alert("이미 기록 대기 중이거나 기록 중입니다.");
    return;
  }

  if (!selectedLabel) {
    alert("라벨을 선택해주세요.");
    return;
  }

  isArmed = true;
  noHandFrameStreak = 0;
  currentRecording = null;

  recordStatus.textContent = `⏺ 대기 중: 손이 인식되면 자동 시작 (${selectedLabel})`;
  recordStatus.classList.remove("rec-active");
  recordStatus.classList.add("rec-idle");
  frameCountStatus.textContent = "프레임: 0";
}

// ── 11. 실제 기록 시작 ───────────────────────────────
function beginActualRecording() {
  currentRecording = {
    id: crypto.randomUUID(),
    label: selectedLabel,
    note: sampleNote.value.trim(),
    createdAt: new Date().toISOString(),
    frames: []
  };

  isRecording = true;
  noHandFrameStreak = 0;

  hudRecord.classList.remove("hidden");
  recordStatus.textContent = `⏺ 기록 중: ${selectedLabel}`;
  recordStatus.classList.add("rec-active");
  recordStatus.classList.remove("rec-idle");
  frameCountStatus.textContent = "프레임: 0";
}

// ── 12. 자동 시작 / 자동 종료 흐름 ───────────────────
function handleAutoRecordingFlow(handResults, faceResults) {
  const hasHand = latestHandCount > 0;

  if (isArmed && !isRecording) {
    if (hasHand) {
      beginActualRecording();
      addFrameToRecording(handResults, faceResults);
    }
    return;
  }

  if (isRecording) {
    if (hasHand) {
      noHandFrameStreak = 0;
      addFrameToRecording(handResults, faceResults);
    } else {
      noHandFrameStreak += 1;
      if (noHandFrameStreak >= NO_HAND_STOP_THRESHOLD) {
        finalizeRecording("손이 내려가 자동 종료");
      }
    }
  }
}

// ── 13. 기록 종료 및 저장 ─────────────────────────────
function finalizeRecording(reasonText = "자동 종료") {
  if (!currentRecording) {
    resetRecordingState();
    return;
  }

  const frameCount = currentRecording.frames.length;

  if (frameCount === 0) {
    resetRecordingState();
    recordStatus.textContent = "⚠️ 프레임 없음 — 다시 시도";
    return;
  }

  collectedSamples.push({ ...currentRecording, frameCount });
  resetRecordingState();
  updateSamplesUI();

  const total = collectedSamples.length;
  recordStatus.textContent = `✅ 저장됨 (${frameCount}f) — 총 ${total}개`;

  if (autoRepeat) {
    setTimeout(() => {
      if (!isArmed && !isRecording) armRecording();
    }, 800);
  }
}

// ── 마지막 표본 삭제 ──────────────────────────────────
function deleteLastSample() {
  if (isArmed || isRecording) {
    alert("기록 중에는 삭제할 수 없습니다.");
    return;
  }
  if (collectedSamples.length === 0) {
    alert("삭제할 표본이 없습니다.");
    return;
  }
  const last = collectedSamples[collectedSamples.length - 1];
  if (!confirm(`마지막 표본을 삭제할까요?\n라벨: ${last.label} / ${last.frameCount}프레임`)) return;
  collectedSamples.pop();
  updateSamplesUI();
  recordStatus.textContent = `↩ 마지막 표본 삭제됨 (총 ${collectedSamples.length}개)`;
}

// ── 14. 수동 종료 버튼 ───────────────────────────────
function stopRecordingManually() {
  if (isRecording && currentRecording) {
    finalizeRecording("수동 종료");
    return;
  }

  if (isArmed && !isRecording) {
    resetRecordingState();
    alert("기록 대기를 취소했습니다.");
    return;
  }

  alert("중지할 기록이 없습니다.");
}

// ── 15. 기록 상태 초기화 ─────────────────────────────
function resetRecordingState() {
  isArmed = false;
  isRecording = false;
  currentRecording = null;
  noHandFrameStreak = 0;

  hudRecord.classList.add("hidden");
  recordStatus.textContent = "⏺ 대기 중";
  recordStatus.classList.remove("rec-active");
  recordStatus.classList.add("rec-idle");
  frameCountStatus.textContent = "프레임: 0";
}

// ── 16. 개별 표본 삭제 ───────────────────────────────
function deleteSampleById(sampleId) {
  if (isArmed || isRecording) {
    alert("기록 대기 중이거나 기록 중에는 표본을 삭제할 수 없습니다.");
    return;
  }

  const targetSample = collectedSamples.find(sample => sample.id === sampleId);
  if (!targetSample) {
    alert("삭제할 표본을 찾을 수 없습니다.");
    return;
  }

  const ok = confirm(
    `이 표본을 삭제할까요?\n라벨: ${targetSample.label}\n프레임 수: ${targetSample.frameCount}`
  );
  if (!ok) return;

  collectedSamples = collectedSamples.filter(sample => sample.id !== sampleId);
  updateSamplesUI();
}

// ── 17. 표본 목록 UI ──────────────────────────────────
function updateSamplesUI() {
  sampleCountStatus.textContent = `저장된 표본: ${collectedSamples.length}개`;

  const summary = {};
  for (const s of collectedSamples) summary[s.label] = (summary[s.label] || 0) + 1;
  labelSummary.innerHTML = Object.entries(summary)
    .map(([label, count]) => `<span class="label-chip">${label} × ${count}</span>`)
    .join("");

  if (collectedSamples.length === 0) {
    samplesList.innerHTML = "아직 저장된 표본이 없습니다.";
    return;
  }

  samplesList.innerHTML = collectedSamples.map(s => `
    <div class="sample-item">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
        <div>
          <div class="s-label">${escapeHtml(s.label)}</div>
          <div class="s-meta">${s.frameCount}f · ${s.note ? escapeHtml(s.note) : "메모없음"}</div>
        </div>
        <button
          class="delete-sample-btn"
          data-sample-id="${s.id}"
          style="background:#dc2626; color:white; border:none; border-radius:8px; padding:6px 10px; cursor:pointer; font-size:12px;"
        >
          삭제
        </button>
      </div>
    </div>
  `).join("");

  const deleteButtons = document.querySelectorAll(".delete-sample-btn");
  deleteButtons.forEach(button => {
    button.addEventListener("click", () => {
      const sampleId = button.dataset.sampleId;
      deleteSampleById(sampleId);
    });
  });
}

// ── 18. JSON 내보내기 ─────────────────────────────────
function exportJson() {
  if (collectedSamples.length === 0) {
    alert("내보낼 표본이 없습니다.");
    return;
  }

  const data = {
    exportedAt: new Date().toISOString(),
    version: "2.0-facemesh",
    situation: situationBadge.textContent,
    totalSamples: collectedSamples.length,
    labelsSummary: getLabelsSummary(),
    samples: collectedSamples
  };

  downloadJson(data, `samples_${Date.now()}.json`);
}

// ── 19. JSON 병합 ─────────────────────────────────────
mergeJsonBtn.addEventListener("click", () => mergeFileInput.click());

mergeFileInput.addEventListener("change", async (e) => {
  const files = Array.from(e.target.files);
  if (files.length === 0) return;

  let merged = [...collectedSamples];
  let addedCount = 0;

  for (const file of files) {
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      const incoming = Array.isArray(json.samples) ? json.samples : [];
      const existingIds = new Set(merged.map(s => s.id));

      for (const s of incoming) {
        if (!existingIds.has(s.id)) {
          merged.push(s);
          existingIds.add(s.id);
          addedCount++;
        }
      }
    } catch (err) {
      console.error(`파일 파싱 실패: ${file.name}`, err);
    }
  }

  collectedSamples = merged;
  updateSamplesUI();
  mergeFileInput.value = "";
  alert(`병합 완료: ${addedCount}개 표본 추가됨 (총 ${collectedSamples.length}개)`);
});

// ── 20. 전체 삭제 ─────────────────────────────────────
function clearSamples() {
  if (isArmed || isRecording) {
    alert("기록 대기 중이거나 기록 중에는 전체 삭제를 할 수 없습니다.");
    return;
  }

  if (!confirm("저장된 모든 표본을 삭제할까요?")) return;
  collectedSamples = [];
  updateSamplesUI();
}

// ── 유틸 ──────────────────────────────────────────────
function getLabelsSummary() {
  const s = {};
  for (const sample of collectedSamples) s[sample.label] = (s[sample.label] || 0) + 1;
  return s;
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;");
}

// ── 이벤트 연결 ───────────────────────────────────────
const autoRepeatBtn = document.getElementById("autoRepeatBtn");
const deleteLastBtn = document.getElementById("deleteLastBtn");

startRecordBtn.addEventListener("click",  armRecording);
stopRecordBtn.addEventListener("click",   stopRecordingManually);
exportJsonBtn.addEventListener("click",   exportJson);
clearSamplesBtn.addEventListener("click", clearSamples);
deleteLastBtn.addEventListener("click",   deleteLastSample);
autoRepeatBtn.addEventListener("click", () => {
  autoRepeat = !autoRepeat;
  autoRepeatBtn.textContent = autoRepeat ? "🔁 연속 모드 ON" : "🔁 연속 모드 OFF";
  autoRepeatBtn.classList.toggle("btn-primary", autoRepeat);
  autoRepeatBtn.classList.toggle("btn-ghost",   !autoRepeat);
});

// ── 앱 시작 ───────────────────────────────────────────
async function initApp() {
  await loadLabels();
  cameraStatus.textContent = "📷 MediaPipe 준비 중...";
  await createLandmarkers();
  await setupCamera();
  resetRecordingState();
  updateSamplesUI();
  predictWebcam();
}

initApp();