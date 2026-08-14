/**
 * Dashboard JavaScript — real-time stats polling + chart updates.
 */

let statsChart = null;
let activityHistory = [];
const MAX_HISTORY = 60;

function startPolling() {
    pollStats();
    setInterval(pollStats, 1000);
    setStatsChart();
}

async function pollStats() {
    try {
        const resp = await fetch('/api/stats');
        if (!resp.ok) return;
        const data = await resp.json();
        updateStatusUI(data);
        addActivityPoint(data);
    } catch (e) {
        // camera might not be open yet — silent fail
    }
}

function updateStatusUI(data) {
    // FPS
    document.getElementById('nav-fps').textContent = `FPS: ${data.fps || '--'}`;

    // Posture
    const pl = (data.posture_label || 'unknown').toUpperCase();
    document.getElementById('posture-display').textContent = pl;

    // Action
    document.getElementById('action-display').textContent =
        (data.action_label || 'unknown').replace(/_/g, ' ').toUpperCase();

    // Threat level
    const pct = Math.round((data.violence_prob || 0) * 100);
    const threatEl = document.getElementById('threat-level');
    threatEl.textContent = `${pct}%`;
    threatEl.style.color = pct > 70 ? '#ef4444' : pct > 40 ? '#f59e0b' : '#10b981';

    // Violence status dot
    const dot = document.getElementById('violence-status');
    if (dot) {
        dot.className = 'status-dot ' + (data.is_violent ? 'danger' : pct > 40 ? 'warning' : 'secure');
    }

    const navSystem = document.getElementById('nav-system-status');
    if (navSystem) {
        navSystem.className = 'status-dot ' + (data.is_violent ? 'danger' : pct > 40 ? 'warning' : 'secure');
    }

    // Alert count
    document.getElementById('alert-count').textContent = data.alert_count;
}

function addActivityPoint(data) {
    activityHistory.push({
        time: new Date().toLocaleTimeString(),
        posture: data.posture_label,
        violence: data.violence_prob || 0,
        fps: data.fps || 0,
    });
    if (activityHistory.length > MAX_HISTORY) activityHistory.shift();
    updateChart();
}

function setStatsChart() {
    const canvas = document.getElementById('activity-chart');
    if (!canvas) return;

    statsChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Violence Probability',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 1.5,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: true,
                },
                {
                    label: 'FPS',
                    data: [],
                    borderColor: '#3b82f6',
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    tension: 0.3,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: {
                    display: true,
                    ticks: { color: '#94a3b8', maxTicksLimit: 10, font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    min: 0, max: 1,
                    ticks: { color: '#94a3b8', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12 }
                }
            }
        }
    });
}

function updateChart() {
    if (!statsChart) return;
    statsChart.data.labels = activityHistory.map(p => p.time);
    statsChart.data.datasets[0].data = activityHistory.map(p => p.violence);
    statsChart.data.datasets[1].data = activityHistory.map(p => p.fps / 30); // normalize to 0-1
    statsChart.update();
}
