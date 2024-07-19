<script setup lang="ts">
import { useCanvasStore } from "@/stores/canvas";

defineProps<{ icon: string; title: string; keys: string[] }>();

const canvasStore = useCanvasStore();

function onClick(element: string) {
  canvasStore.displayKey(element);
}

function onItemDrag(event: DragEvent, key: string) {
  console.log(`onDrag: ${key}`);
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = "copy";
    event.dataTransfer.effectAllowed = "copy";
    console.log(`onItemDrag: ${key}`);
    event.dataTransfer.setData("key", key);
  }
}
</script>

<template>
  <h2><span :class="icon"></span>{{ title }}</h2>
  <div class="keys">
    <div
      v-for="key in keys"
      :key="key"
      class="key"
      @click="onClick(key)"
      draggable="true"
      @dragstart="onItemDrag($event, key)"
    >
      {{ key }}
    </div>
  </div>
</template>

<style scoped>
h2 {
  color: var(--primary-color);
}

.keys {
  display: flex;
  box-sizing: border-box;
  flex: none;
  flex-direction: column;
  flex-grow: 1;
  align-items: flex-start;
  align-self: stretch;
  padding: 16px 24px 24px;
  border-width: 1px 1px 0 0;
  border-style: solid;
  border-color: var(--border-color);
  background: var(--background-color);
  gap: 10px;
}
</style>
