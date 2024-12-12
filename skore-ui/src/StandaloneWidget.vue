<script setup lang="ts">
import MediaWidgetSelector from "./components/MediaWidgetSelector.vue";
import { deserializeProjectItemDto } from "./models";

const dataId = `skore-item-data-${window.skoreWidgetId}`;
const itemData = JSON.parse(document.getElementById(dataId)?.innerText || "{}");
const item = deserializeProjectItemDto(itemData);
console.info(itemData);
</script>

<template>
  <div class="widget">
    <MediaWidgetSelector :item="item" />
  </div>
</template>

<style scoped>
.widget {
  width: 100%;
}
</style>
