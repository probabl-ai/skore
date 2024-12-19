<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";

import MediaWidgetSelector from "@/components/MediaWidgetSelector.vue";
import { deserializeProjectItemDto } from "@/models";
import { isUserInDarkMode } from "@/services/utils";

const dataId = `skore-item-data-${window.skoreWidgetId}`;
const itemData = JSON.parse(document.getElementById(dataId)?.innerText || "{}");
const item = deserializeProjectItemDto(itemData);
const theme = ref(isUserInDarkMode() ? "skore-dark" : "skore-light");

function switchTheme(m: MediaQueryListEvent) {
  theme.value = m.matches ? "skore-dark" : "skore-light";
}

const preferredColorScheme = window.matchMedia("(prefers-color-scheme: dark)");

onMounted(() => {
  preferredColorScheme.addEventListener("change", switchTheme);
});

onBeforeUnmount(() => {
  preferredColorScheme.removeEventListener("change", switchTheme);
});
</script>

<template>
  <div class="widget" :class="theme">
    <MediaWidgetSelector :item="item" />
  </div>
</template>

<style scoped>
.widget {
  width: 100%;
  background-color: var(--color-background-primary);
}
</style>
