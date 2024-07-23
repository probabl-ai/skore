<script setup lang="ts">
import { ref } from "vue";

import { useCanvasStore } from "@/stores/canvas";

const props = defineProps<{ itemKey: string; metadata: string }>();
const canvasStore = useCanvasStore();
const isDraggable = ref(false);

function onClick() {
  canvasStore.displayKey(props.itemKey);
}

function onDragStart(event: DragEvent) {
  if (event.dataTransfer) {
    event.dataTransfer.setData("key", props.itemKey);
  }
}
</script>

<template>
  <div
    class="key"
    @click="onClick"
    :draggable="isDraggable"
    @dragstart="onDragStart($event)"
    @mouseup="isDraggable = false"
    @mouseleave="isDraggable = false"
  >
    <div class="handle" @mousedown="isDraggable = true">=</div>
    <div class="label">
      {{ props.itemKey }}
    </div>
    <div class="metadata">Added unknown #FIXME</div>
  </div>
</template>

<style scoped>
.key {
  position: relative;
  width: 100%;
  padding: var(--spacing-padding-small);
  border: var(--border-color-elevated) 1px solid;
  border-radius: var(--border-radius);
  cursor: pointer;
  opacity: 1;
  transition: opacity var(--transition-duration) var(--transition-easing);

  &[draggable="true"] {
    opacity: 0.5;
  }

  & .handle {
    position: absolute;
    top: 0;
    right: 5px;
    color: var(--text-color-normal);
    cursor: move;
  }

  & .label {
    color: var(--text-color-highlight);
    font-size: var(--text-size-highlight);
    font-weight: var(--text-weight-highlight);
  }

  & .metadata {
    color: var(--text-color-normal);
    font-size: var(--text-size-normal);
    font-weight: var(--text-weight-normal);
  }
}
</style>
