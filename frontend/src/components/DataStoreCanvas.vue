<script setup lang="ts">
import { useCanvasStore } from "@/stores/canvas";
import { computed } from "vue";

const canvasStore = useCanvasStore();

const elements = computed(() => {
  return canvasStore.displayedKeys;
});
</script>

<template>
  <div class="widgets">
    <div v-for="(element, index) in elements" :key="index" class="canvas-element">
      {{ element }}
    </div>
  </div>
</template>

<style scoped>
.widgets {
  display: flex;
  overflow: scroll;
  flex-direction: column;
  padding: 10px;
}
</style>
