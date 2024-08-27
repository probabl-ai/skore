<script setup lang="ts">
import { computed, onBeforeMount, ref } from "vue";
import { useRouter } from "vue-router";

import { type FileTreeNode } from "@/components/FileTree.vue";
import { remap, sha1 } from "@/services/utils";
import { useReportsStore } from "@/stores/reports";

const router = useRouter();
const props = defineProps<FileTreeNode>();
const reportsStore = useReportsStore();

const randomColor = ref("");
const hasChildren = computed(() => props.children?.length);
const label = computed(() => {
  const segment = props.uri.split("/");
  return segment[segment.length - 1];
});
const isSelected = computed(() => {
  return reportsStore.selectedReportUri === props.uri;
});

function onClick() {
  var segments = props.uri.split("/");

  // Take out leading slash to get a valid URL
  const [head, ...tail] = segments;
  if (head === "") {
    segments = tail;
  }

  router.push({
    name: "dashboard",
    params: {
      segments: segments,
    },
  });
}

onBeforeMount(async () => {
  const hash = await sha1(props.uri);
  const hashAsNumber = Number(`0x${hash.slice(0, 7)}`);
  const hue = remap(hashAsNumber, 0x0000000, 0xfffffff, 0, 360);
  randomColor.value = `background-color: hsl(${hue}deg 97 75);`;
});
</script>

<template>
  <div class="file-tree-item" :style="`--indentation-level: ${indentationLevel};`">
    <div class="label">
      <div v-if="props.indentationLevel == 0" class="top-level-indicator" :style="randomColor" />
      <div v-else class="child-indicator">L</div>
      <div class="text" :class="{ selected: isSelected }" @click="onClick">
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

    & .text.selected,
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
