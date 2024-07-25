<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import { type FileTreeNode } from "./FileTree.vue";

const router = useRouter();
const props = defineProps<FileTreeNode>();

const hasChildren = computed(() => props.children?.length);
const label = computed(() => {
  const segment = props.uri.split("/");
  return segment[segment.length - 1];
});

const randomColor = computed(() => {
  const hue = Math.random() * 360;
  return `background-color: hsl(${hue}deg 97 75);`;
});

function onClick() {
  router.push({
    name: "dashboard",
    params: {
      segments: props.uri.split("/"),
    },
  });
}
</script>

<template>
  <div class="file-tree-item" :style="`--indentation-level: ${indentationLevel};`">
    <div class="label">
      <div v-if="props.indentationLevel == 0" class="top-level-indicator" :style="randomColor" />
      <div v-else class="child-indicator">L</div>
      <div class="text" @click="onClick">
        {{ label }}
      </div>
    </div>
    <div class="children" v-if="hasChildren">
      <FileTreeItem
        v-for="(child, index) in props.children"
        :key="index"
        :uri="child.uri"
        :children="child.children"
        :indentation-level="(props.indentationLevel ?? 0) + 1"
      />
    </div>
  </div>
</template>

<style scoped>
.file-tree-item {
  margin-left: calc(8px * var(--indentation-level));

  .label {
    display: flex;
    height: 28px;
    flex-direction: row;
    align-items: center;
    color: var(--text-color-normal);
    cursor: pointer;
    font-size: var(--text-size-title);
    font-weight: var(--text-weight-normal);
    gap: var(--spacing-gap-small);

    & .text {
      padding: 1px 8px;
      border-radius: 3px;
      transition:
        color var(--transition-duration) var(--transition-easing),
        background-color var(--transition-duration) var(--transition-easing);
    }

    & .text:hover {
      background-color: var(--background-color-selected);
      color: var(--text-color-highlight);
    }

    & .top-level-indicator {
      width: 4px;
      height: 4px;
      border-radius: 0.5px;
    }

    & .child-indicator {
      font-size: 0.8em;
    }
  }
}
</style>
