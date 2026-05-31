const SVG_NS = "http://www.w3.org/2000/svg";

function middleEllipsis(text, head = 5, tail = 5) {
    if (text.length <= head + tail + 3) {
        return text;
    }
    return text.slice(0, head) + "..." + text.slice(-tail);
}

function formatDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }
    const pad = (n) => String(n).padStart(2, "0");
    return (
        date.getUTCFullYear() +
        "-" +
        pad(date.getUTCMonth() + 1) +
        "-" +
        pad(date.getUTCDate()) +
        " " +
        pad(date.getUTCHours()) +
        ":" +
        pad(date.getUTCMinutes()) +
        ":" +
        pad(date.getUTCSeconds())
    );
}

function formatNumber(raw) {
    if (!raw) {
        return "";
    }
    const value = Number.parseFloat(raw);
    if (!Number.isFinite(value)) {
        return "";
    }
    return Number.parseFloat(value.toPrecision(6)).toString();
}

function pad(value) {
    return String(value).padStart(2, "0");
}

function formatBucket(date, unit) {
    const day =
        date.getUTCFullYear() +
        "-" +
        pad(date.getUTCMonth() + 1) +
        "-" +
        pad(date.getUTCDate());
    if (unit === "second") {
        return (
            day +
            " " +
            pad(date.getUTCHours()) +
            ":" +
            pad(date.getUTCMinutes()) +
            ":" +
            pad(date.getUTCSeconds())
        );
    }
    if (unit === "minute") {
        return day + " " + pad(date.getUTCHours()) + ":" + pad(date.getUTCMinutes());
    }
    if (unit === "hour") {
        return day + " " + pad(date.getUTCHours()) + ":00";
    }
    if (unit === "month") {
        return date.getUTCFullYear() + "-" + pad(date.getUTCMonth() + 1);
    }
    return day;
}

function customBucketLabel(date, count, unit) {
    const step = Math.max(1, count);
    if (unit === "month") {
        const months = date.getUTCFullYear() * 12 + date.getUTCMonth();
        const bucket = Math.floor(months / step) * step;
        const start = new Date(Date.UTC(Math.floor(bucket / 12), bucket % 12, 1));
        return formatBucket(start, "month");
    }
    const unitMs = {
        second: 1000,
        minute: 60000,
        hour: 3600000,
        day: 86400000,
        week: 604800000,
    }[unit];
    const buckets = Math.floor(date.getTime() / unitMs);
    const start = new Date(Math.floor(buckets / step) * step * unitMs);
    return formatBucket(start, unit);
}

function compareValues(a, b, kind) {
    const emptyA = a === "";
    const emptyB = b === "";
    if (emptyA || emptyB) {
        // Always push empty values to the bottom.
        return emptyA === emptyB ? 0 : emptyA ? 1 : -1;
    }
    if (kind === "number") {
        return parseFloat(a) - parseFloat(b);
    }
    if (kind === "date") {
        return new Date(a) - new Date(b);
    }
    return a < b ? -1 : a > b ? 1 : 0;
}

function anchorFor(i, count) {
    if (i === 0) return "start";
    if (i === count - 1) return "end";
    return "middle";
}

function formatTick(value) {
    if (!isFinite(value)) {
        return "";
    }
    return Number.parseFloat(value.toPrecision(4)).toString();
}

function colorFor(t) {
    // Map t in [0, 1] across a blue -> light gray -> red gradient; NaN is gray.
    if (Number.isNaN(t)) {
        return "rgb(150, 150, 150)";
    }
    const stops = [
        [59, 76, 192],
        [221, 221, 221],
        [180, 4, 38],
    ];
    const clamped = Math.min(1, Math.max(0, t));
    const scaled = clamped * (stops.length - 1);
    const lower = Math.floor(scaled);
    const upper = Math.min(stops.length - 1, lower + 1);
    const frac = scaled - lower;
    const mix = (a, b) => Math.round(a + (b - a) * frac);
    return (
        "rgb(" +
        mix(stops[lower][0], stops[upper][0]) +
        ", " +
        mix(stops[lower][1], stops[upper][1]) +
        ", " +
        mix(stops[lower][2], stops[upper][2]) +
        ")"
    );
}

function appendText(group, className, x, y, anchor, text) {
    const node = document.createElementNS(SVG_NS, "text");
    node.setAttribute("class", className);
    node.setAttribute("x", x);
    node.setAttribute("y", y);
    node.setAttribute("text-anchor", anchor);
    node.textContent = text;
    group.appendChild(node);
}

function appendLine(group, x1, y1, x2, y2, stroke, className) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("class", className);
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("stroke", stroke);
    group.appendChild(line);
}
