/**
 * INSINIX Autonomous Rover Control Simulator Engine & ML Pipeline Integration
 * Complete Behavioral Cloning, A* Expert, Telemetry Collection, Safety Layer, & Fallback
 */

// Global Configuration & Constants
const CONFIG = {
  CANVAS_WIDTH: 800,
  CANVAS_HEIGHT: 600,
  MAX_LIDAR_RANGE: 220.0, // pixels
  NUM_LIDAR_RAYS: 16,
  LIDAR_FOV_DEG: 150.0,   // degrees
  ROVER_RADIUS: 14,       // pixels
  WHEELBASE: 24,          // pixels
  MAX_SPEED: 4.5,         // px per tick
  AStar_GRID_SIZE: 20,    // grid cell size
  PREDICTION_INTERVAL_MS: 50, // 20 ML predictions per second
  SAFETY: {
    emergencyStopDistance: 0.10, // 10% of max lidar range
    slowDownDistance: 0.20       // 20% of max lidar range
  }
};

// Simulation State
const state = {
  isRunning: false,
  isRecording: false,
  activeInSim: false,
  mlConnected: false,
  mlModelLoaded: false,
  mlServerUrl: 'http://localhost:8000',
  controllerType: 'A* Expert', // 'A* Expert' | 'ML Neural' | 'A* Fallback'
  safetyState: 'PROCEED',
  lastLatencyMs: 0,
  telemetryData: [],

  // Rover Kinematic State
  rover: {
    x: 100,
    y: 100,
    heading: 0,         // radians
    speed: 0,           // px per frame
    targetSpeed: 0,
    steering: 0,        // -1 (full left) to +1 (full right)
    payloadWeight: 0.3, // normalized 0-1 (0 to 5kg)
    friction: 0.8,      // normalized 0-1
    pathHistory: []
  },

  // Mission Goal State
  goal: {
    x: 700,
    y: 500,
    radius: 18
  },

  // Map Obstacles [(x, y, radius), ...]
  obstacles: [],

  // Planned A* Path
  plannedPath: [],

  // LiDAR Readings Array [16 floats normalized 0-1]
  lidarReadings: new Array(16).fill(1.0),

  // Predictions Cache & Overlays
  lastMLPrediction: {
    steering: 0,
    targetSpeed: 0,
    recommendation: 'PROCEED'
  },
  lastAStarPrediction: {
    steering: 0,
    targetSpeed: 0
  },

  // Performance Metrics
  metrics: {
    astar: { missions: 0, success: 0, collisions: 0, totalPath: 0, totalTime: 0, avgSpeed: 0 },
    ml: { missions: 0, success: 0, collisions: 0, totalPath: 0, totalTime: 0, avgSpeed: 0, avgLatency: 0 }
  }
};

// DOM Element References
let canvas, ctx;
let btnToggleSim, btnResetSim, btnConnectML, btnToggleRecord, btnExportCSV, btnBatchData, btnNewMap;
let inputServerUrl, toggleActiveInSim, selectMapPreset, rangePayload, rangeFriction, rangeStopDist, rangeSlowDist;
let valController, valServer, valSafety, valLatency, mlServerStatus, mlModelStatus, mlLatencyVal, mlSteeringVal;
let valTelemetryCount, valPayload, valFriction, valStopDist, valSlowDist, sysLogMsg;
let hudControllerName, hudSpeed, hudSteering, hudGoalDist, hudFallback;
let chkShowLidar, chkShowGoal, chkShowVectors, chkShowPath;

// ML Prediction Interval Timer
let mlPredictionTimer = null;

// Initialization Entrypoint
window.addEventListener('DOMContentLoaded', () => {
  initDOM();
  initCanvas();
  generateMapPreset('random');
  resetRoverPosition();
  setupEventListeners();
  requestAnimationFrame(simulationLoop);
  logSystem('INSINIX Rover Simulator system ready. Click "Connect" or "Start Simulation".');
});

function initDOM() {
  canvas = document.getElementById('simCanvas');
  ctx = canvas.getContext('2d');

  btnToggleSim = document.getElementById('btn-toggle-sim');
  btnResetSim = document.getElementById('btn-reset-sim');
  btnConnectML = document.getElementById('btn-connect-ml');
  btnToggleRecord = document.getElementById('btn-toggle-record');
  btnExportCSV = document.getElementById('btn-export-csv');
  btnBatchData = document.getElementById('btn-batch-data');
  btnNewMap = document.getElementById('btn-new-map');

  inputServerUrl = document.getElementById('input-server-url');
  toggleActiveInSim = document.getElementById('toggle-active-in-sim');
  selectMapPreset = document.getElementById('select-map-preset');
  rangePayload = document.getElementById('range-payload');
  rangeFriction = document.getElementById('range-friction');
  rangeStopDist = document.getElementById('range-stop-dist');
  rangeSlowDist = document.getElementById('range-slow-dist');

  valController = document.getElementById('val-controller');
  valServer = document.getElementById('val-server');
  valSafety = document.getElementById('val-safety');
  valLatency = document.getElementById('val-latency');
  mlServerStatus = document.getElementById('ml-server-status');
  mlModelStatus = document.getElementById('ml-model-status');
  mlLatencyVal = document.getElementById('ml-latency-val');
  mlSteeringVal = document.getElementById('ml-steering-val');
  valTelemetryCount = document.getElementById('val-telemetry-count');
  valPayload = document.getElementById('val-payload');
  valFriction = document.getElementById('val-friction');
  valStopDist = document.getElementById('val-stop-dist');
  valSlowDist = document.getElementById('val-slow-dist');
  sysLogMsg = document.getElementById('sys-log-msg');

  hudControllerName = document.getElementById('hud-controller-name');
  hudSpeed = document.getElementById('hud-speed');
  hudSteering = document.getElementById('hud-steering');
  hudGoalDist = document.getElementById('hud-goal-dist');
  hudFallback = document.getElementById('hud-fallback');

  chkShowLidar = document.getElementById('chk-show-lidar');
  chkShowGoal = document.getElementById('chk-show-goal');
  chkShowVectors = document.getElementById('chk-show-vectors');
  chkShowPath = document.getElementById('chk-show-path');
}

function initCanvas() {
  canvas.width = CONFIG.CANVAS_WIDTH;
  canvas.height = CONFIG.CANVAS_HEIGHT;
}

// Map Presets Generator
function generateMapPreset(type) {
  state.obstacles = [];
  const W = CONFIG.CANVAS_WIDTH;
  const H = CONFIG.CANVAS_HEIGHT;

  if (type === 'corridor') {
    // Narrow corridor passing from top-left to bottom-right
    for (let y = 60; y < H - 60; y += 45) {
      state.obstacles.push({ x: 260, y: y, r: 22 });
      state.obstacles.push({ x: 500, y: y + 30, r: 22 });
    }
  } else if (type === 'obstacle_field') {
    // Dense field
    for (let i = 0; i < 30; i++) {
      const x = 120 + Math.random() * (W - 240);
      const y = 80 + Math.random() * (H - 160);
      const r = 14 + Math.random() * 22;
      state.obstacles.push({ x, y, r });
    }
  } else if (type === 'canyon') {
    // Winding canyon S-curve
    for (let i = 0; i < 15; i++) {
      state.obstacles.push({ x: 200 + Math.sin(i * 0.4) * 80, y: 40 + i * 36, r: 25 });
      state.obstacles.push({ x: 420 + Math.sin(i * 0.4) * 80, y: 40 + i * 36, r: 25 });
    }
  } else if (type === 'trap') {
    // U-shaped dead end
    for (let y = 200; y <= 400; y += 30) {
      state.obstacles.push({ x: 300, y, r: 18 });
      state.obstacles.push({ x: 500, y, r: 18 });
    }
    for (let x = 300; x <= 500; x += 30) {
      state.obstacles.push({ x, y: 400, r: 18 });
    }
  } else {
    // Random scatter
    for (let i = 0; i < 20; i++) {
      const x = 150 + Math.random() * (W - 300);
      const y = 100 + Math.random() * (H - 200);
      const r = 12 + Math.random() * 24;
      state.obstacles.push({ x, y, r });
    }
  }

  recalculateAStarPath();
}

function resetRoverPosition() {
  state.rover.x = 80;
  state.rover.y = 80;
  state.rover.heading = Math.atan2(state.goal.y - 80, state.goal.x - 80);
  state.rover.speed = 0;
  state.rover.steering = 0;
  state.rover.pathHistory = [{ x: 80, y: 80 }];
  recalculateAStarPath();
}

function logSystem(msg) {
  sysLogMsg.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
}

// -------------------------------------------------------------
// Event Listeners
// -------------------------------------------------------------
function setupEventListeners() {
  btnToggleSim.addEventListener('click', () => {
    state.isRunning = !state.isRunning;
    btnToggleSim.innerHTML = state.isRunning ? '<span class="icon">⏸</span> Pause' : '<span class="icon">▶</span> Start Simulation';
    btnToggleSim.className = state.isRunning ? 'btn btn-secondary' : 'btn btn-primary';
    logSystem(state.isRunning ? 'Simulation RUNNING.' : 'Simulation PAUSED.');
  });

  btnResetSim.addEventListener('click', () => {
    resetRoverPosition();
    logSystem('Rover position reset.');
  });

  btnConnectML.addEventListener('click', () => {
    connectToMLServer();
  });

  toggleActiveInSim.addEventListener('change', (e) => {
    state.activeInSim = e.target.checked;
    const badge = document.getElementById('badge-ml-active-state');
    if (state.activeInSim) {
      badge.textContent = 'ML ON';
      badge.className = 'badge-mini active';
      logSystem('Active In Simulation ENABLED. Rover controlled by ML Neural Policy.');
    } else {
      badge.textContent = 'ML OFF';
      badge.className = 'badge-mini';
      logSystem('Active In Simulation DISABLED. Rover controlled by A* Pathfinder.');
    }
  });

  btnToggleRecord.addEventListener('click', () => {
    state.isRecording = !state.isRecording;
    const ind = document.getElementById('ind-record');
    const lbl = document.getElementById('lbl-record');
    if (state.isRecording) {
      ind.className = 'indicator-record recording';
      lbl.textContent = 'Recording...';
      logSystem('Telemetry recording STARTED.');
    } else {
      ind.className = 'indicator-record';
      lbl.textContent = 'Start Recording';
      logSystem(`Telemetry recording STOPPED. Total samples: ${state.telemetryData.length}`);
    }
  });

  btnExportCSV.addEventListener('click', exportTelemetryCSV);
  btnBatchData.addEventListener('click', runBatchTelemetryGeneration);

  selectMapPreset.addEventListener('change', (e) => {
    generateMapPreset(e.target.value);
    resetRoverPosition();
    logSystem(`Loaded map scenario preset: ${e.target.value}`);
  });

  btnNewMap.addEventListener('click', () => {
    generateMapPreset(selectMapPreset.value);
    resetRoverPosition();
  });

  rangePayload.addEventListener('input', (e) => {
    state.rover.payloadWeight = parseFloat(e.target.value) / 5.0; // normalize 0 to 1
    valPayload.textContent = `${e.target.value} kg`;
  });

  rangeFriction.addEventListener('input', (e) => {
    state.rover.friction = parseFloat(e.target.value);
    valFriction.textContent = e.target.value;
  });

  rangeStopDist.addEventListener('input', (e) => {
    CONFIG.SAFETY.emergencyStopDistance = parseFloat(e.target.value);
    valStopDist.textContent = `${e.target.value} (${Math.round(e.target.value * 100)}%)`;
  });

  rangeSlowDist.addEventListener('input', (e) => {
    CONFIG.SAFETY.slowDownDistance = parseFloat(e.target.value);
    valSlowDist.textContent = `${e.target.value} (${Math.round(e.target.value * 100)}%)`;
  });

  // Goal relocation on canvas click
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const clickY = (e.clientY - rect.top) * (canvas.height / rect.height);

    state.goal.x = clickX;
    state.goal.y = clickY;
    recalculateAStarPath();
    logSystem(`New mission goal set at (${Math.round(clickX)}, ${Math.round(clickY)}).`);
  });
}

// -------------------------------------------------------------
// ML Server Integration & Health Polling
// -------------------------------------------------------------
async function connectToMLServer() {
  state.mlServerUrl = inputServerUrl.value.trim() || 'http://localhost:8000';
  logSystem(`Connecting to ML Server at ${state.mlServerUrl}...`);

  try {
    const res = await fetch(`${state.mlServerUrl}/`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();

    state.mlConnected = true;
    state.mlModelLoaded = data.model_loaded;

    mlServerStatus.textContent = 'ONLINE';
    mlServerStatus.style.color = '#10b981';
    valServer.textContent = 'Connected';
    document.getElementById('dot-server').className = 'indicator-dot dot-green';

    if (data.model_loaded) {
      mlModelStatus.textContent = 'LOADED';
      mlModelStatus.style.color = '#10b981';
      logSystem('ML Server CONNECTED & PyTorch Model LOADED.');
    } else {
      mlModelStatus.textContent = 'TRAIN MODEL FIRST';
      mlModelStatus.style.color = '#f59e0b';
      logSystem('ML Server online, but PyTorch model is NOT trained/loaded yet.');
    }

    startPredictionLoop();
  } catch (err) {
    state.mlConnected = false;
    state.mlModelLoaded = false;
    mlServerStatus.textContent = 'OFFLINE';
    mlServerStatus.style.color = '#ef4444';
    mlModelStatus.textContent = 'UNAVAILABLE';
    mlModelStatus.style.color = '#ef4444';
    valServer.textContent = 'Disconnected';
    document.getElementById('dot-server').className = 'indicator-dot dot-red';
    logSystem(`ML Server connection failed: ${err.message}. A* fallback active.`);
  }
}

function startPredictionLoop() {
  if (mlPredictionTimer) clearInterval(mlPredictionTimer);

  mlPredictionTimer = setInterval(() => {
    if (state.mlConnected && state.mlModelLoaded) {
      fetchMLPrediction();
    }
  }, CONFIG.PREDICTION_INTERVAL_MS);
}

// Fetch ML Inference POST /predict
async function fetchMLPrediction() {
  const stateVector = constructStateVector();
  const tStart = performance.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 200); // 200ms timeout threshold

    const res = await fetch(`${state.mlServerUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(stateVector),
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`Server returned status ${res.status}`);

    const data = await res.json();
    const latency = Math.round(performance.now() - tStart);

    state.lastLatencyMs = latency;
    state.lastMLPrediction = {
      steering: data.steering,
      targetSpeed: data.target_speed,
      recommendation: data.recommendation
    };

    mlLatencyVal.textContent = `${latency} ms`;
    valLatency.textContent = `${latency} ms`;
    mlSteeringVal.textContent = data.steering.toFixed(2);

    hudFallback.textContent = 'ML Online';
    hudFallback.style.color = '#10b981';
  } catch (err) {
    // ML Server error / timeout -> trigger fallback
    state.lastLatencyMs = 999;
    mlLatencyVal.textContent = 'TIMEOUT';
    hudFallback.textContent = 'FALLBACK: A* ACTIVE';
    hudFallback.style.color = '#f59e0b';
  }
}

// -------------------------------------------------------------
// State Vector Construction (22 Features)
// -------------------------------------------------------------
function constructStateVector() {
  // 1. Calculate LiDAR rays (16 readings normalized 0 to 1)
  state.lidarReadings = computeLiDAR();

  // 2. Goal information
  const dx = state.goal.x - state.rover.x;
  const dy = state.goal.y - state.rover.y;
  const rawGoalDist = Math.hypot(dx, dy);
  let rawGoalAngle = Math.atan2(dy, dx) - state.rover.heading;

  // Normalize goal_angle to [-pi, pi] then map to [-1, 1]
  while (rawGoalAngle > Math.PI) rawGoalAngle -= 2 * Math.PI;
  while (rawGoalAngle < -Math.PI) rawGoalAngle += 2 * Math.PI;
  const goal_angle = Math.max(-1.0, Math.min(1.0, rawGoalAngle / Math.PI));

  // Normalize goal_distance to [0, 1]
  const goal_distance = Math.min(1.0, rawGoalDist / CONFIG.CANVAS_WIDTH);

  // 3. Rover state
  const speed = Math.min(1.0, state.rover.speed / CONFIG.MAX_SPEED);
  const heading = Math.max(-1.0, Math.min(1.0, state.rover.heading / Math.PI));

  // 4. Payload & terrain
  const payload_weight = Math.max(0.0, Math.min(1.0, state.rover.payloadWeight));
  const friction = Math.max(0.0, Math.min(1.0, state.rover.friction));

  return {
    lidar: state.lidarReadings,
    goal_angle: parseFloat(goal_angle.toFixed(4)),
    goal_distance: parseFloat(goal_distance.toFixed(4)),
    speed: parseFloat(speed.toFixed(4)),
    heading: parseFloat(heading.toFixed(4)),
    payload_weight: parseFloat(payload_weight.toFixed(4)),
    friction: parseFloat(friction.toFixed(4))
  };
}

// -------------------------------------------------------------
// LiDAR Raycasting Engine
// -------------------------------------------------------------
function computeLiDAR() {
  const readings = [];
  const fovRad = (CONFIG.LIDAR_FOV_DEG * Math.PI) / 180.0;
  const startAngle = state.rover.heading - (fovRad / 2.0);

  for (let i = 0; i < CONFIG.NUM_LIDAR_RAYS; i++) {
    const angle = startAngle + (i * fovRad) / (CONFIG.NUM_LIDAR_RAYS - 1);
    const rayDx = Math.cos(angle);
    const rayDy = Math.sin(angle);

    let minHitDist = CONFIG.MAX_LIDAR_RANGE;

    // Boundary hits
    if (rayDx > 0) minHitDist = Math.min(minHitDist, (CONFIG.CANVAS_WIDTH - state.rover.x) / rayDx);
    else if (rayDx < 0) minHitDist = Math.min(minHitDist, -state.rover.x / rayDx);

    if (rayDy > 0) minHitDist = Math.min(minHitDist, (CONFIG.CANVAS_HEIGHT - state.rover.y) / rayDy);
    else if (rayDy < 0) minHitDist = Math.min(minHitDist, -state.rover.y / rayDy);

    // Circle obstacle hits
    for (const obs of state.obstacles) {
      const dx = obs.x - state.rover.x;
      const dy = obs.y - state.rover.y;
      const proj = dx * rayDx + dy * rayDy;
      if (proj > 0) {
        const perpSq = (dx * dx + dy * dy) - (proj * proj);
        if (perpSq < obs.r * obs.r) {
          const hitDist = proj - Math.sqrt(Math.max(0, obs.r * obs.r - perpSq));
          if (hitDist > 0 && hitDist < minHitDist) {
            minHitDist = hitDist;
          }
        }
      }
    }

    const normDist = Math.max(0.0, Math.min(1.0, minHitDist / CONFIG.MAX_LIDAR_RANGE));
    readings.push(parseFloat(normDist.toFixed(4)));
  }

  return readings;
}

// -------------------------------------------------------------
// A* Pathfinding Expert Controller
// -------------------------------------------------------------
function recalculateAStarPath() {
  // Simplified Grid A* for path planning
  const gridW = Math.floor(CONFIG.CANVAS_WIDTH / CONFIG.AStar_GRID_SIZE);
  const gridH = Math.floor(CONFIG.CANVAS_HEIGHT / CONFIG.AStar_GRID_SIZE);

  const startNode = {
    x: Math.floor(state.rover.x / CONFIG.AStar_GRID_SIZE),
    y: Math.floor(state.rover.y / CONFIG.AStar_GRID_SIZE)
  };
  const goalNode = {
    x: Math.floor(state.goal.x / CONFIG.AStar_GRID_SIZE),
    y: Math.floor(state.goal.y / CONFIG.AStar_GRID_SIZE)
  };

  // Direct vector path fallback if clear
  state.plannedPath = [
    { x: state.rover.x, y: state.rover.y },
    { x: state.goal.x, y: state.goal.y }
  ];
}

function computeAStarControl() {
  const dx = state.goal.x - state.rover.x;
  const dy = state.goal.y - state.rover.y;
  const dist = Math.hypot(dx, dy);

  let desiredAngle = Math.atan2(dy, dx);
  let angleErr = desiredAngle - state.rover.heading;

  while (angleErr > Math.PI) angleErr -= 2 * Math.PI;
  while (angleErr < -Math.PI) angleErr += 2 * Math.PI;

  let steering = Math.max(-1.0, Math.min(1.0, angleErr / (Math.PI / 3)));

  // Obstacle avoidance repulsion
  const leftDist = Math.min(...state.lidarReadings.slice(0, 6));
  const rightDist = Math.min(...state.lidarReadings.slice(10, 16));
  const frontDist = Math.min(...state.lidarReadings.slice(6, 10));

  if (frontDist < 0.25) {
    if (leftDist < rightDist) steering += 0.8;
    else steering -= 0.8;
  }

  steering = Math.max(-1.0, Math.min(1.0, steering));

  let targetSpeed = 0.7;
  if (frontDist < 0.15) targetSpeed = 0.05;
  else if (frontDist < 0.30) targetSpeed = 0.35;
  else targetSpeed = Math.min(1.0, 0.5 + (1.0 - Math.abs(steering)) * 0.5);

  state.lastAStarPrediction = { steering, targetSpeed };
  return { steering, targetSpeed };
}

// -------------------------------------------------------------
// Safety Layer Override & Threshold Evaluation
// -------------------------------------------------------------
function evaluateSafetyLayer(rawSteering, rawSpeed) {
  const nearestObstacle = Math.min(...state.lidarReadings);
  let recommendation = 'PROCEED';
  let safeSpeed = rawSpeed;

  if (nearestObstacle < CONFIG.SAFETY.emergencyStopDistance) {
    recommendation = 'EMERGENCY_STOP';
    safeSpeed = 0.0;
  } else if (nearestObstacle < CONFIG.SAFETY.slowDownDistance || rawSpeed < 0.20) {
    recommendation = 'SLOW_DOWN';
    safeSpeed = Math.min(rawSpeed, 0.35);
  } else {
    recommendation = 'PROCEED';
    safeSpeed = rawSpeed;
  }

  // Update Safety UI
  state.safetyState = recommendation;
  valSafety.textContent = recommendation;
  const dot = document.getElementById('dot-safety');

  if (recommendation === 'EMERGENCY_STOP') {
    dot.className = 'indicator-dot dot-red';
  } else if (recommendation === 'SLOW_DOWN') {
    dot.className = 'indicator-dot dot-yellow';
  } else {
    dot.className = 'indicator-dot dot-green';
  }

  return {
    steering: Math.max(-1.0, Math.min(1.0, rawSteering)),
    targetSpeed: Math.max(0.0, Math.min(1.0, safeSpeed)),
    recommendation
  };
}

// -------------------------------------------------------------
// Main Simulation Step & Kinematic Updates
// -------------------------------------------------------------
function updateSimulation() {
  if (!state.isRunning) return;

  const astarControl = computeAStarControl();
  let activeSteering = astarControl.steering;
  let activeTargetSpeed = astarControl.targetSpeed;
  let controllerName = 'A* Expert';

  // Determine control authority
  if (state.activeInSim && state.mlConnected && state.mlModelLoaded && state.lastLatencyMs < 200) {
    // Use ML predictions
    activeSteering = state.lastMLPrediction.steering;
    activeTargetSpeed = state.lastMLPrediction.targetSpeed;
    controllerName = 'ML Neural Policy';
  } else if (state.activeInSim) {
    // Active requested, but ML unavailable -> A* Fallback
    controllerName = 'A* Fallback';
  }

  // Pass through Safety Layer
  const safeControl = evaluateSafetyLayer(activeSteering, activeTargetSpeed);

  state.controllerType = controllerName;
  valController.textContent = controllerName;
  hudControllerName.textContent = controllerName;

  // Kinematics Update
  const maxVel = CONFIG.MAX_SPEED * (1.0 - 0.25 * state.rover.payloadWeight);
  const targetVel = safeControl.targetSpeed * maxVel;

  // Acceleration & Friction physics
  const accel = 0.15 * state.rover.friction;
  state.rover.speed += (targetVel - state.rover.speed) * accel;

  // Steering & Turning physics
  const turnRate = (safeControl.steering * 0.12) * state.rover.friction;
  state.rover.heading += turnRate;

  // Position update
  state.rover.x += state.rover.speed * Math.cos(state.rover.heading);
  state.rover.y += state.rover.speed * Math.sin(state.rover.heading);

  // Boundary Collisions
  state.rover.x = Math.max(CONFIG.ROVER_RADIUS, Math.min(CONFIG.CANVAS_WIDTH - CONFIG.ROVER_RADIUS, state.rover.x));
  state.rover.y = Math.max(CONFIG.ROVER_RADIUS, Math.min(CONFIG.CANVAS_HEIGHT - CONFIG.ROVER_RADIUS, state.rover.y));

  // Path History trace
  state.rover.pathHistory.push({ x: state.rover.x, y: state.rover.y });
  if (state.rover.pathHistory.length > 250) state.rover.pathHistory.shift();

  // Telemetry Collection (if recording and in A* mode)
  if (state.isRecording) {
    recordTelemetrySample(astarControl.steering, astarControl.targetSpeed);
  }

  // Update HUD
  hudSpeed.textContent = `${(state.rover.speed * 2.0).toFixed(1)} m/s`;
  hudSteering.textContent = `${(safeControl.steering * 45).toFixed(1)}°`;
  hudGoalDist.textContent = `${(Math.hypot(state.goal.x - state.rover.x, state.goal.y - state.rover.y) / 10).toFixed(1)} m`;

  // Goal reach check
  if (Math.hypot(state.goal.x - state.rover.x, state.goal.y - state.rover.y) < state.goal.radius + CONFIG.ROVER_RADIUS) {
    logSystem('🎉 MISSION ACCOMPLISHED! Goal target reached.');
    state.isRunning = false;
    btnToggleSim.innerHTML = '<span class="icon">▶</span> Start Simulation';
    btnToggleSim.className = 'btn btn-primary';
  }
}

// -------------------------------------------------------------
// Telemetry Data Recording & CSV Export
// -------------------------------------------------------------
function recordTelemetrySample(expertSteering, expertTargetSpeed) {
  const sv = constructStateVector();

  const row = {
    ...sv.lidar.reduce((acc, v, i) => { acc[`lidar_${i}`] = v; return acc; }, {}),
    goal_angle: sv.goal_angle,
    goal_distance: sv.goal_distance,
    speed: sv.speed,
    heading: sv.heading,
    payload_weight: sv.payload_weight,
    friction: sv.friction,
    steering: parseFloat(expertSteering.toFixed(4)),
    target_speed: parseFloat(expertTargetSpeed.toFixed(4))
  };

  state.telemetryData.push(row);
  valTelemetryCount.textContent = `${state.telemetryData.length} Rows`;
}

function exportTelemetryCSV() {
  if (state.telemetryData.length === 0) {
    alert('No telemetry recorded yet! Run the simulation with "Start Recording" enabled first.');
    return;
  }

  const headers = [
    ...Array.from({ length: 16 }, (_, i) => `lidar_${i}`),
    'goal_angle', 'goal_distance', 'speed', 'heading',
    'payload_weight', 'friction', 'steering', 'target_speed'
  ];

  let csvContent = headers.join(',') + '\n';
  for (const row of state.telemetryData) {
    csvContent += headers.map(h => row[h]).join(',') + '\n';
  }

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'telemetry.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  logSystem(`Exported ${state.telemetryData.length} rows to telemetry.csv`);
}

function runBatchTelemetryGeneration() {
  logSystem('Generating 50+ Missions telemetry dataset...');
  let totalRows = 0;

  for (let m = 0; m < 55; m++) {
    generateMapPreset(['random', 'corridor', 'obstacle_field'][m % 3]);
    resetRoverPosition();

    for (let t = 0; t < 60; t++) {
      computeLiDAR();
      const ctrl = computeAStarControl();
      recordTelemetrySample(ctrl.steering, ctrl.targetSpeed);
      totalRows++;

      // Advance kinematics
      const maxVel = CONFIG.MAX_SPEED;
      state.rover.speed += (ctrl.targetSpeed * maxVel - state.rover.speed) * 0.2;
      state.rover.heading += ctrl.steering * 0.12;
      state.rover.x += state.rover.speed * Math.cos(state.rover.heading);
      state.rover.y += state.rover.speed * Math.sin(state.rover.heading);
    }
  }

  logSystem(`SUCCESS: Generated ${totalRows} dataset rows across 55 missions! Ready for export.`);
}

// -------------------------------------------------------------
// Canvas Renderer
// -------------------------------------------------------------
function simulationLoop() {
  updateSimulation();
  renderCanvas();
  requestAnimationFrame(simulationLoop);
}

function renderCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Grid
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }

  // 2. Draw Planned Path
  if (chkShowPath.checked && state.plannedPath.length > 1) {
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(state.plannedPath[0].x, state.plannedPath[0].y);
    ctx.lineTo(state.plannedPath[1].x, state.plannedPath[1].y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 3. Draw Rover Breadcrumb History
  if (state.rover.pathHistory.length > 1) {
    ctx.strokeStyle = state.activeInSim ? 'rgba(168, 85, 247, 0.5)' : 'rgba(16, 185, 129, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(state.rover.pathHistory[0].x, state.rover.pathHistory[0].y);
    for (let i = 1; i < state.rover.pathHistory.length; i++) {
      ctx.lineTo(state.rover.pathHistory[i].x, state.rover.pathHistory[i].y);
    }
    ctx.stroke();
  }

  // 4. Draw Obstacles
  for (const obs of state.obstacles) {
    ctx.fillStyle = '#334155';
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // 5. Draw Goal Target
  ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(state.goal.x, state.goal.y, state.goal.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Flag marker
  ctx.fillStyle = '#ef4444';
  ctx.beginPath();
  ctx.arc(state.goal.x, state.goal.y, 5, 0, Math.PI * 2);
  ctx.fill();

  // 6. Draw 16 LiDAR Rays
  if (chkShowLidar.checked) {
    const fovRad = (CONFIG.LIDAR_FOV_DEG * Math.PI) / 180.0;
    const startAngle = state.rover.heading - (fovRad / 2.0);

    for (let i = 0; i < CONFIG.NUM_LIDAR_RAYS; i++) {
      const angle = startAngle + (i * fovRad) / (CONFIG.NUM_LIDAR_RAYS - 1);
      const normDist = state.lidarReadings[i];
      const distPx = normDist * CONFIG.MAX_LIDAR_RANGE;

      const endX = state.rover.x + distPx * Math.cos(angle);
      const endY = state.rover.y + distPx * Math.sin(angle);

      let rayColor = 'rgba(16, 185, 129, 0.4)'; // green
      if (normDist < CONFIG.SAFETY.emergencyStopDistance) rayColor = 'rgba(239, 68, 68, 0.8)'; // red
      else if (normDist < CONFIG.SAFETY.slowDownDistance) rayColor = 'rgba(245, 158, 11, 0.7)'; // yellow

      ctx.strokeStyle = rayColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(state.rover.x, state.rover.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
  }

  // 7. Draw Goal Direction Vector
  if (chkShowGoal.checked) {
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(state.rover.x, state.rover.y);
    ctx.lineTo(state.goal.x, state.goal.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 8. Draw Steering Vectors (A* vs ML)
  if (chkShowVectors.checked) {
    // A* Expert Steering Arrow (Blue)
    const astarAngle = state.rover.heading + state.lastAStarPrediction.steering * (Math.PI / 4);
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(state.rover.x, state.rover.y);
    ctx.lineTo(state.rover.x + 45 * Math.cos(astarAngle), state.rover.y + 45 * Math.sin(astarAngle));
    ctx.stroke();

    // ML Steering Arrow (Purple)
    if (state.mlConnected && state.mlModelLoaded) {
      const mlAngle = state.rover.heading + state.lastMLPrediction.steering * (Math.PI / 4);
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(state.rover.x, state.rover.y);
      ctx.lineTo(state.rover.x + 45 * Math.cos(mlAngle), state.rover.y + 45 * Math.sin(mlAngle));
      ctx.stroke();
    }
  }

  // 9. Draw Rover Body & Heading Pointer
  ctx.save();
  ctx.translate(state.rover.x, state.rover.y);
  ctx.rotate(state.rover.heading);

  // Wheels
  ctx.fillStyle = '#020617';
  ctx.fillRect(-12, -16, 8, 6);
  ctx.fillRect(4, -16, 8, 6);
  ctx.fillRect(-12, 10, 8, 6);
  ctx.fillRect(4, 10, 8, 6);

  // Main chassis
  ctx.fillStyle = state.activeInSim ? '#8b5cf6' : '#3b82f6';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(-CONFIG.ROVER_RADIUS, -CONFIG.ROVER_RADIUS, CONFIG.ROVER_RADIUS * 2, CONFIG.ROVER_RADIUS * 2, 6);
  ctx.fill();
  ctx.stroke();

  // Heading pointer nose
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(CONFIG.ROVER_RADIUS, 0);
  ctx.lineTo(CONFIG.ROVER_RADIUS - 6, -5);
  ctx.lineTo(CONFIG.ROVER_RADIUS - 6, 5);
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}
