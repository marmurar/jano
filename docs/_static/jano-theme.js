(function () {
  const storageKey = "jano-docs-theme";
  const root = document.documentElement;

  function systemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function storedTheme() {
    try {
      const theme = window.localStorage.getItem(storageKey);
      return theme === "dark" || theme === "light" ? theme : null;
    } catch (_) {
      return null;
    }
  }

  function setStoredTheme(theme) {
    try {
      window.localStorage.setItem(storageKey, theme);
    } catch (_) {
      return;
    }
  }

  function applyTheme(theme) {
    root.dataset.janoTheme = theme;
    root.style.colorScheme = theme;

    const toggle = document.querySelector("[data-theme-toggle]");
    if (!toggle) {
      return;
    }

    const isDark = theme === "dark";
    const darkLabel = toggle.dataset.darkLabel || "Dark mode";
    const lightLabel = toggle.dataset.lightLabel || "Light mode";
    const darkTitle = toggle.dataset.darkTitle || darkLabel;
    const lightTitle = toggle.dataset.lightTitle || lightLabel;
    toggle.setAttribute("aria-pressed", String(isDark));
    toggle.textContent = isDark ? lightLabel : darkLabel;
    toggle.title = isDark ? lightTitle : darkTitle;
  }

  applyTheme(storedTheme() || systemTheme());

  document.addEventListener("DOMContentLoaded", function () {
    applyTheme(root.dataset.janoTheme || storedTheme() || systemTheme());

    const toggle = document.querySelector("[data-theme-toggle]");
    if (toggle) {
      toggle.addEventListener("click", function () {
        const nextTheme = root.dataset.janoTheme === "dark" ? "light" : "dark";
        setStoredTheme(nextTheme);
        applyTheme(nextTheme);
      });
    }

    if (window.matchMedia) {
      const media = window.matchMedia("(prefers-color-scheme: dark)");
      media.addEventListener("change", function () {
        if (!storedTheme()) {
          applyTheme(systemTheme());
        }
      });
    }
  });
})();
