<script setup lang="ts">
import { useCanvasStore } from "@/stores/canvas";

const canvasStore = useCanvasStore();

function onItemDrop(event: DragEvent) {
  if (event.dataTransfer) {
    const key = event.dataTransfer.getData("key");
    console.log(`onItemDrop: ${key}`);
    canvasStore.displayKey(key);
  }
}
</script>

<template>
  <div class="canvas" @drop="onItemDrop($event)" @dragover.prevent>
    <div
      v-for="key in canvasStore.displayedKeys"
      :key="key"
      class="canvas-element"
      v-html="canvasStore.get(key)"
    ></div>
  </div>
</template>

<style scoped>
.canvas {
  display: flex;
  overflow: scroll;
  flex-direction: column;
  flex-grow: 1;
  padding: 10px;

  & .canvas-element {
    max-width: 100%;
  }
}
</style>
