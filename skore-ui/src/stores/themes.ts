import { acceptHMRUpdate, defineStore } from "pinia";
import { computed, ref } from "vue";

import { isUserInDarkMode } from "@/services/utils";

export type Theme = "light" | "dark";

export const useThemesStore = defineStore("themes", () => {
  const currentTheme = ref<Theme>(isUserInDarkMode() ? "dark" : "light");
  const currentThemeClass = computed(() => `skore-${currentTheme.value}`);

  /**
   * Set the theme globbaly.
   * @param t the theme to set
   */
  function setTheme(t: Theme) {
    console.log("set theme", t);
    currentTheme.value = t;
  }

  // listen to browser preferences changes
  const preferredColorScheme = window.matchMedia("(prefers-color-scheme: dark)");
  preferredColorScheme.addEventListener("change", (m) => {
    setTheme(m.matches ? "dark" : "light");
  });

  return { currentTheme, currentThemeClass, setTheme };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useThemesStore, import.meta.hot));
}
