<script setup lang="ts">
import { onMounted, ref, watch } from "vue";

import FloatingTooltip from "@/components/FloatingTooltip.vue";
import SlideToggle from "@/components/SlideToggle.vue";
import SkoreLogo from "@/components/icons/SkoreLogo.vue";
import { getInfo } from "@/services/api";
import { useThemesStore } from "@/stores/themes";

const themesStore = useThemesStore();
const isDarkModeForced = ref(themesStore.currentTheme === "dark");
const projectName = ref("");
const projectPath = ref("");

watch(isDarkModeForced, (forceDarkMode) => {
  themesStore.setTheme(forceDarkMode ? "dark" : "light");
});
watch(
  () => themesStore.currentTheme,
  () => {
    isDarkModeForced.value = themesStore.currentTheme === "dark";
  }
);

onMounted(async () => {
  const info = await getInfo();
  if (info) {
    projectName.value = info.name;
    projectPath.value = info.path;
  }
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
    <div class="project-name">
      <FloatingTooltip :text="projectPath" placement="bottom-start">
        {{ projectName }}
      </FloatingTooltip>
    </div>
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
  width: 100%;
  height: var(--height-header);
  flex-direction: row;
  border-right: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  background-color: var(--color-background-secondary);

  & .logo {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-8);
    border-right: solid var(--stroke-width-md) var(--color-stroke-background-primary);

    & svg {
      height: calc(var(--height-header) * 0.8);
    }
  }

  & .project-name {
    display: flex;
    flex: 1;
    align-items: center;
    color: var(--color-text-primary);
  }

  & nav {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: var(--spacing-8);
    gap: var(--spacing-8);
  }

  & .tools {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-8);
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
