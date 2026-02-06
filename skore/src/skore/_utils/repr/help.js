function detectTheme(refElementId) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a reference element's color (luma)
    if (refElementId) {
        const refElement = document.getElementById(refElementId);
        if (refElement) {
            const color = window.getComputedStyle(refElement, null).getPropertyValue('color');
            const match = color.match(/^rgb[a]?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
            if (match) {
                const r = parseInt(match[1], 10);
                const g = parseInt(match[2], 10);
                const b = parseInt(match[3], 10);
                // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
                const luma = 0.299 * r + 0.587 * g + 0.114 * b;

                if (luma > 180) {
                    return 'dark';
                }
                if (luma < 75) {
                    return 'light';
                }
            }
        }
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyThemeToHelpContainer(container, refElementId) {
    const theme = detectTheme(refElementId);
    if (theme === 'dark') {
        container.setAttribute('data-theme', 'dark');
        container.classList.add('dark-theme');
    } else {
        container.removeAttribute('data-theme');
        container.classList.remove('dark-theme');
    }
}
