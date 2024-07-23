<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";

import DashboardHeader from "@/components/DashboardHeader.vue";
import DataStoreCanvas from "@/components/DataStoreCanvas.vue";
import DataStoreKeyList from "@/components/DataStoreKeyList.vue";
import FileTree, { transformUrisToTree } from "@/components/FileTree.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import { type DataStore } from "@/models";
import { fetchAllManderUris, fetchMander } from "@/services/api";
import { useCanvasStore } from "@/stores/canvas";

const route = useRoute();
const dataStoreUris = ref<string[]>([]);
const dataStore = ref<DataStore | null>();
const canvasStore = useCanvasStore();
const isDropIndicatorVisible = ref(false);
const editor = ref<HTMLDivElement>();
const editorHeader = ref<HTMLElement>();
const fileTree = computed(() => transformUrisToTree(dataStoreUris.value));
const editorHeaderHeight = computed(() =>
  editorHeader.value ? editorHeader.value.clientHeight : 0
);

async function fetchDataStoreDetail(path: string | string[]) {
  const p = Array.isArray(path) ? path.join("/") : path;
  if (p.length == 0) {
    return;
  }
  const m = await fetchMander(p);
  dataStore.value = m;
  canvasStore.setDataStore(m);
}

function onSaveBoard(/*event: PointerEvent*/) {
  alert("not implemented yet");
}

function onItemDrop(event: DragEvent) {
  isDropIndicatorVisible.value = false;
  if (event.dataTransfer) {
    const key = event.dataTransfer.getData("key");
    canvasStore.displayKey(key);
  }
}

function onDragEnter() {
  isDropIndicatorVisible.value = true;
}

function onDragLeave(event: DragEvent) {
  if (event.target == editor.value) {
    isDropIndicatorVisible.value = false;
  }
}

watch(
  () => route.params.segments,
  async (newSegments) => {
    await fetchDataStoreDetail(newSegments);
  }
);

dataStoreUris.value = await fetchAllManderUris();
await fetchDataStoreDetail(route.params.segments);
</script>

<template>
  <main>
    <nav>
      <DashboardHeader title="File Manager" icon="icon-folder" />
      <FileTree :nodes="fileTree" />
    </nav>
    <article :class="{ 'not-found': dataStore == null }" v-if="dataStore">
      <div class="elements">
        <DashboardHeader title="Elements (added from mandr)" icon="icon-pie-chart" />
        <div class="key-list">
          <DataStoreKeyList title="Views" icon="icon-plot" :keys="Object.keys(dataStore.views)" />
          <DataStoreKeyList title="Info" icon="icon-text" :keys="Object.keys(dataStore.info)" />
          <DataStoreKeyList title="Logs" icon="icon-folder" :keys="Object.keys(dataStore.logs)" />
          <DataStoreKeyList
            title="Artifacts"
            icon="icon-gift"
            :keys="Object.keys(dataStore.artifacts)"
          />
        </div>
      </div>
      <div
        ref="editor"
        class="editor"
        @drop="onItemDrop"
        @dragover.prevent
        @dragenter="onDragEnter"
        @dragleave="onDragLeave"
      >
        <div class="editor-header" ref="editorHeader">
          <h1>Board</h1>
          <SimpleButton label="Save report settings" @click="onSaveBoard" />
        </div>
        <div class="drop-indicator" :class="{ visible: isDropIndicatorVisible }"></div>
        <Transition name="fade">
          <div
            v-if="!isDropIndicatorVisible && canvasStore.displayedKeys.length === 0"
            class="placeholder"
            :style="`--header-height: ${editorHeaderHeight}px`"
          >
            No item selected yet, start by dragging one element!
          </div>
        </Transition>
        <DataStoreCanvas />
      </div>
    </article>
    <div v-else>mandr not found...</div>
  </main>
</template>

<style scoped>
main {
  display: flex;
  flex-direction: row;

  nav,
  article {
    height: 100dvh;
  }

  nav {
    display: flex;
    width: 240px;
    flex-direction: column;
    flex-shrink: 0;
    border-right: solid 1px var(--border-color-normal);

    & .file-tree {
      overflow: scroll;
      flex-grow: 1;
    }
  }

  article {
    display: flex;
    flex-direction: row;
    flex-grow: 1;

    &.not-found {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #c8c7c7;
      font-size: x-large;
      font-weight: 200;
    }

    & .elements {
      display: flex;
      width: 240px;
      height: 100dvh;
      flex-direction: column;
      flex-shrink: 0;
      border-right: solid 1px var(--border-color-normal);

      & .key-list {
        overflow: scroll;
        flex-grow: 1;
        background-color: var(--background-color-normal);
      }
    }

    & .editor {
      display: flex;
      flex-direction: column;
      flex-grow: 1;

      & .placeholder {
        height: 100%;
        padding-top: calc((100vh - var(--header-height)) * 476 / 730);
        background-image: url("../assets/images/editor-placeholder.png");
        background-position: 50%;
        background-repeat: no-repeat;
        background-size: contain;
        color: var(--text-color-normal);
        font-size: var(--text-size-normal);
        text-align: center;
      }

      & .editor-header {
        display: flex;
        height: 44px;
        align-items: center;
        padding: var(--spacing-padding-large);
        border-right: solid var(--border-width-normal) var(--border-color-normal);
        border-bottom: solid var(--border-width-normal) var(--border-color-normal);
        background-color: var(--background-color-elevated);

        & h1 {
          flex-grow: 1;
          color: var(--text-color-normal);
          font-size: var(--text-size-title);
          font-weight: var(--text-weight-title);
          text-align: center;
        }
      }

      & .drop-indicator {
        height: 3px;
        border-radius: 8px;
        margin: 0 10%;
        background-color: var(--text-color-title);
        opacity: 0;
        transition: opacity var(--transition-duration) var(--transition-easing);

        &.visible {
          opacity: 1;
        }
      }
    }
  }
}
</style>
