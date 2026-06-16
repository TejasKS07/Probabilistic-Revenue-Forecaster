/**
 * charts.js — Chart.js configuration for light-mode enterprise dashboard
 */

// -- Global Chart.js defaults (light mode) --
Chart.defaults.color = '#5c6370';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.animation.duration = 600;
Chart.defaults.animation.easing = 'easeOutQuart';

const COLORS = {
    primary: '#1a1f36',
    blue: '#3b5bdb',
    teal: '#0ca678',
    orange: '#e8590c',
    purple: '#7c3aed',
    rose: '#e03131',
    slate: '#64748b',

    blueLight: 'rgba(59, 91, 219, 0.12)',
    tealLight: 'rgba(12, 166, 120, 0.12)',
    orangeLight: 'rgba(232, 89, 12, 0.12)',
    purpleLight: 'rgba(124, 58, 237, 0.12)',
};

const CHANNEL_STYLES = {
    google: { color: '#3b5bdb', light: 'rgba(59, 91, 219, 0.12)', name: 'Google Ads' },
    meta:   { color: '#7c3aed', light: 'rgba(124, 58, 237, 0.12)', name: 'Meta Ads' },
    bing:   { color: '#0ca678', light: 'rgba(12, 166, 120, 0.12)', name: 'Microsoft Ads' },
};

function formatCurrency(value) {
    if (Math.abs(value) >= 1_000_000) return '$' + (value / 1_000_000).toFixed(1) + 'M';
    if (Math.abs(value) >= 1_000) return '$' + (value / 1_000).toFixed(0) + 'k';
    return '$' + value.toFixed(0);
}

function formatRoas(value) {
    return value.toFixed(1) + 'x';
}

const GRID_LIGHT = {
    color: 'rgba(0, 0, 0, 0.06)',
    drawBorder: false,
};

/**
 * Revenue forecast line chart with confidence band
 */
function createRevenueLineChart(ctx, labels, datasets) {
    return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                tooltip: {
                    backgroundColor: '#1a1f36',
                    titleColor: '#fff',
                    bodyColor: '#e5e7eb',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.dataset.label && ctx.dataset.label.includes('band')) return null;
                            return `${ctx.dataset.label}: ${formatCurrency(ctx.raw)}`;
                        },
                    },
                    filter: (item) => item.dataset.label && !item.dataset.label.includes('band'),
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        filter: (item) => item.text && !item.text.includes('band'),
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false, drawBorder: false },
                    border: { dash: [4, 4] },
                },
                y: {
                    grid: { ...GRID_LIGHT, drawBorder: false },
                    border: { display: false },
                    ticks: { callback: (v) => formatCurrency(v) },
                    beginAtZero: true,
                },
            },
        },
    });
}

/**
 * ROAS forecast line chart
 */
function createRoasLineChart(ctx, labels, datasets) {
    return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                tooltip: {
                    backgroundColor: '#1a1f36',
                    titleColor: '#fff',
                    bodyColor: '#e5e7eb',
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.dataset.label && ctx.dataset.label.includes('band')) return null;
                            return `${ctx.dataset.label}: ${formatRoas(ctx.raw)}`;
                        },
                    },
                    filter: (item) => item.dataset.label && !item.dataset.label.includes('band'),
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        filter: (item) => item.text && !item.text.includes('band'),
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    border: { dash: [4, 4] },
                },
                y: {
                    grid: { ...GRID_LIGHT, drawBorder: false },
                    border: { display: false },
                    ticks: { callback: (v) => formatRoas(v) },
                    beginAtZero: true,
                },
            },
        },
    });
}

window.ChartFactory = {
    createRevenueLineChart,
    createRoasLineChart,
    formatCurrency,
    formatRoas,
    COLORS,
    CHANNEL_STYLES,
};
