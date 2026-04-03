/**
 * ComparisonReport: shadow shell + theme; Results show full comparison metrics;
 * Estimator / Data each have a synced <select>; only one light-DOM pair uses
 * slot="estimator-display" / slot="table-report" at a time.
 */
function skoreInitComparisonReport(containerId) {
    const wrapper = document.getElementById(containerId + "-wrapper");
    const host = document.getElementById(containerId);
    if (!wrapper || !host) {
        return;
    }

    const template = document.getElementById(containerId + "-template");
    if (!template) {
        return;
    }

    const shadowRoot = host.attachShadow({ mode: "closed" });
    shadowRoot.appendChild(template.content.cloneNode(true));

    const root = shadowRoot.querySelector(".container");
    if (root) {
        applyThemeToReportContainer(root, containerId + "-wrapper");
    }

    shadowRoot.querySelectorAll(".report-disclosure .tooltip-text a").forEach((a) => {
        a.addEventListener("click", (e) => {
            e.stopPropagation();
        });
    });

    const selects = shadowRoot.querySelectorAll(".skore-comparison-report-select");
    if (selects.length === 0) {
        return;
    }

    function setActiveComparisonReport(index) {
        const idx = String(index);

        selects.forEach((sel) => {
            sel.value = idx;
        });

        host.querySelectorAll(".skore-cmp-estimator-slot").forEach((el) => {
            if (el.dataset.comparisonIndex === idx) {
                el.setAttribute("slot", "estimator-display");
            } else {
                el.removeAttribute("slot");
            }
        });

        host.querySelectorAll(".skore-cmp-table-slot").forEach((el) => {
            if (el.dataset.comparisonIndex === idx) {
                el.setAttribute("slot", "table-report");
            } else {
                el.removeAttribute("slot");
            }
        });

        host.querySelectorAll(".skore-cmp-diagnostic-slot").forEach((el) => {
            if (el.dataset.comparisonIndex === idx) {
                el.setAttribute("slot", "diagnostic");
            } else {
                el.removeAttribute("slot");
            }
        });
    }

    selects.forEach((sel) => {
        sel.addEventListener("change", () => {
            setActiveComparisonReport(sel.value);
        });
    });

    setActiveComparisonReport(selects[0].value);
}
