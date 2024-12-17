<script setup lang="ts">
import { ref, watch } from "vue";

import SlideToggle from "@/components/SlideToggle.vue";
import SkoreLogo from "@/components/icons/SkoreLogo.vue";
import { useThemesStore } from "@/stores/themes";

const { setTheme, currentTheme } = useThemesStore();
const isDarkModeForced = ref(currentTheme === "dark");

watch(isDarkModeForced, (forceDarkMode) => {
  setTheme(forceDarkMode ? "dark" : "light");
});
</script>

<template>
  <div class="app-toolbar">
    <div class="logo">
      <SkoreLogo />
    </div>
    <nav>
      <slot></slot>
    </nav>
    <div class="tools">
      <div class="dark-light">
        <i class="icon icon-sun" />
        <SlideToggle v-model:is-toggled="isDarkModeForced" />
        <i class="icon icon-moon" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.app-toolbar {
  display: flex;
  min-width: var(--width-toolbar);
  height: 100dvh;
  flex-direction: column;
  border-right: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  background-color: var(--color-background-secondary);

  & .logo {
    display: flex;
    height: var(--height-header);
    align-items: center;
    justify-content: center;
    border-bottom: solid var(--stroke-width-md) var(--color-stroke-background-primary);

    & svg {
      width: calc(var(--width-toolbar) * 0.8);
    }
  }

  & nav {
    display: flex;
    flex: 1;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing-12) 0;
    gap: var(--spacing-10);
  }

  & .tools {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-8) 0;
    gap: var(--spacing-8);

    & .dark-light {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      gap: var(--spacing-4);

      & .icon {
        color: var(--color-text-primary);
        font-size: var(--font-size-xs);
      }
    }
  }
}
</style>
