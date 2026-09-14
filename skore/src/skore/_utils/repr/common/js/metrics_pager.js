function skoreInitMetricsPagers(root) {
    root.querySelectorAll(".skore-metrics-pager").forEach(skoreInitMetricsPager);
}

function skoreInitMetricsPager(pager) {
    const rows = Array.from(pager.querySelectorAll("tbody tr"));
    if (rows.length === 0) {
        return;
    }

    pager.classList.add("is-ready");
    let nPages = 0;
    rows.forEach((row) => {
        nPages = Math.max(nPages, Number(row.dataset.page) + 1);
    });

    const status = pager.querySelector(".skore-metrics-pager-status");
    const first = pager.querySelector(".skore-metrics-pager-first");
    const prev = pager.querySelector(".skore-metrics-pager-prev");
    const next = pager.querySelector(".skore-metrics-pager-next");
    const last = pager.querySelector(".skore-metrics-pager-last");
    let page = 0;

    function render() {
        let start = rows.length;
        let end = 0;
        rows.forEach((row, index) => {
            const onPage = Number(row.dataset.page) === page;
            row.classList.toggle("is-active", onPage);
            if (onPage) {
                start = Math.min(start, index + 1);
                end = Math.max(end, index + 1);
            }
        });
        status.textContent = "Results: " + start + "-" + end + " of " + rows.length;
        const atStart = page === 0;
        const atEnd = page === nPages - 1;
        first.disabled = atStart;
        prev.disabled = atStart;
        next.disabled = atEnd;
        last.disabled = atEnd;
    }

    first.addEventListener("click", () => {
        page = 0;
        render();
    });
    prev.addEventListener("click", () => {
        page = Math.max(0, page - 1);
        render();
    });
    next.addEventListener("click", () => {
        page = Math.min(nPages - 1, page + 1);
        render();
    });
    last.addEventListener("click", () => {
        page = nPages - 1;
        render();
    });
    render();
}
