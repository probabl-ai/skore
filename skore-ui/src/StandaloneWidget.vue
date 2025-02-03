<script setup lang="ts">
import { onBeforeUnmount } from "vue";

import MediaWidgetSelector from "@/components/MediaWidgetSelector.vue";
import { deserializeProjectItemDto } from "@/models";
import { useThemesStore } from "@/stores/themes";

const themesStore = useThemesStore();
const dataId = `skore-item-data-${window.skoreWidgetId}`;
const itemData = JSON.parse(document.getElementById(dataId)?.innerText || "{}");
const item = deserializeProjectItemDto(itemData);

onBeforeUnmount(() => {
  themesStore.dispose();
});
</script>

<template>
  <div class="widget" :class="themesStore.currentThemeClass">
    <MediaWidgetSelector :item="item" />
  </div>
</template>

<style scoped>
.widget {
  width: 100%;
  background-color: var(--color-background-primary);
  color: var(--color-text-primary);
}
</style>
