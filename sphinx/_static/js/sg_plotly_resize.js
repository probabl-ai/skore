function resizePlotlyGraphs() {
    // Find all plotly graph divs
    const plotlyDivs = document.getElementsByClassName("plotly-graph-div");

    // Iterate through each div and resize
    for (const div of plotlyDivs) {
        Plotly.Plots.resize(div);
    }
}

// Call resize when window is resized
window.addEventListener("resize", resizePlotlyGraphs);

// Initial resize call after plots are created
document.addEventListener("DOMContentLoaded", resizePlotlyGraphs);
