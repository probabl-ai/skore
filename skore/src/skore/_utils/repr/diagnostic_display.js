function skoreInitDiagnosticDisplay(containerId) {
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

    shadowRoot.querySelectorAll(".report-tabset .tooltip-text a").forEach((a) => {
        a.addEventListener("click", (e) => {
            e.stopPropagation();
        });
    });
}
