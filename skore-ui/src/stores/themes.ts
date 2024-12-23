import { acceptHMRUpdate, defineStore } from "pinia";
import { computed, ref } from "vue";

import { isUserInDarkMode } from "@/services/utils";

export type Theme = "light" | "dark";

const LOCAL_STORAGE_KEY = "skore-theme";

export const useThemesStore = defineStore("themes", () => {
  const preferredTheme = isUserInDarkMode() ? "dark" : "light";
  const storedTheme = localStorage.getItem(LOCAL_STORAGE_KEY);
  const currentTheme = ref<Theme>((storedTheme ?? preferredTheme) as Theme);
  const currentThemeClass = computed(() => `skore-${currentTheme.value}`);

  /**
   * Set the theme globbaly.
   * @param t the theme to set
   */
  function setTheme(t: Theme) {
    currentTheme.value = t;
    localStorage.setItem(LOCAL_STORAGE_KEY, t);
  }

  // listen to browser preferences changes
  function onColorSchemeChange(m: MediaQueryListEvent) {
    setTheme(m.matches ? "dark" : "light");
  }
  const preferredColorScheme = window.matchMedia("(prefers-color-scheme: dark)");
  preferredColorScheme.addEventListener("change", onColorSchemeChange);

  function dispose() {
    preferredColorScheme.removeEventListener("change", onColorSchemeChange);
  }

  return { currentTheme, currentThemeClass, setTheme, dispose };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useThemesStore, import.meta.hot));
}
