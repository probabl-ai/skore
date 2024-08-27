<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";

import DashboardHeader from "@/components/DashboardHeader.vue";
import DataStoreCanvas from "@/components/DataStoreCanvas.vue";
import DataStoreKeyList from "@/components/DataStoreKeyList.vue";
import FileTree, { transformUrisToTree } from "@/components/FileTree.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import { fetchShareableBlob } from "@/services/api";
import { saveBlob } from "@/services/utils";
import { useReportsStore } from "@/stores/reports";

const route = useRoute();
const reportsStore = useReportsStore();
const isDropIndicatorVisible = ref(false);
const editor = ref<HTMLDivElement>();
const fileTree = computed(() => transformUrisToTree(reportsStore.reportUris));

async function onShareReport(/*event: PointerEvent*/) {
  const uri = reportsStore.selectedReport?.uri;
  if (uri) {
    const shareable = await fetchShareableBlob(uri);
    if (shareable) {
      saveBlob(
        shareable,
        uri.replace(/\//g, (m, i) => {
          return i === 0 ? "" : "-";
        })
      );
    }
  }
}

function onItemDrop(event: DragEvent) {
  isDropIndicatorVisible.value = false;
  if (event.dataTransfer) {
    const key = event.dataTransfer.getData("key");
    reportsStore.displayKey(key);
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
  (newSegments) => {
    const uri = Array.isArray(newSegments) ? newSegments.join("/") : newSegments;
    reportsStore.selectedReportUri = uri;
    // relaunch the background polling to get report right now
    reportsStore.stopBackendPolling();
    reportsStore.startBackendPolling();
  }
);

onMounted(() => reportsStore.startBackendPolling());
onBeforeUnmount(() => reportsStore.stopBackendPolling());
</script>

<template>
  <main>
    <nav>
      <DashboardHeader title="File Manager" icon="icon-folder" />
      <Simplebar class="file-trees">
        <FileTree :nodes="fileTree" />
      </Simplebar>
    </nav>
    <article v-if="reportsStore.selectedReport">
      <div class="elements">
        <DashboardHeader title="Elements (added from mandr)" icon="icon-pie-chart" />
        <Simplebar class="key-list">
          <DataStoreKeyList
            title="Plots"
            icon="icon-plot"
            :keys="reportsStore.selectedReport.plotKeys"
          />
          <DataStoreKeyList
            title="Info"
            icon="icon-text"
            :keys="reportsStore.selectedReport.infoKeys"
          />
        </Simplebar>
      </div>

      <div
        ref="editor"
        class="editor"
        @drop="onItemDrop"
        @dragover.prevent
        @dragenter="onDragEnter"
        @dragleave="onDragLeave"
      >
        <div class="editor-header">
          <h1>Report</h1>
          <SimpleButton label="Share report" @click="onShareReport" />
        </div>
        <div class="drop-indicator" :class="{ visible: isDropIndicatorVisible }"></div>
        <Transition name="fade">
          <div
            v-if="!isDropIndicatorVisible && reportsStore.layout.length === 0"
            class="placeholder"
          >
            <div class="wrapper">No item selected yet, start by dragging one element!</div>
          </div>
          <Simplebar class="canvas-wrapper" v-else>
            <DataStoreCanvas />
          </Simplebar>
        </Transition>
      </div>
    </article>
    <div class="not-found" v-else>mandr not found...</div>
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

    & .file-trees {
      height: 0;
      flex-grow: 1;
    }
  }

  article,
  .not-found {
    display: flex;
    flex-direction: row;
    flex-grow: 1;
  }

  article {
    & .elements {
      display: flex;
      width: 240px;
      flex-direction: column;
      flex-shrink: 0;
      border-right: solid 1px var(--border-color-normal);

      & .key-list {
        height: 0;
        flex-grow: 1;
        background-color: var(--background-color-normal);
      }
    }

    & .editor {
      display: flex;
      max-height: 100vh;
      flex-direction: column;
      flex-grow: 1;

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
        height: 0;
        border-radius: 8px;
        margin: 0 10%;
        background-color: var(--text-color-title);
        opacity: 0;
        transition: opacity var(--transition-duration) var(--transition-easing);

        &.visible {
          height: 3px;
          opacity: 1;
        }
      }

      & .placeholder {
        display: flex;
        height: 100%;
        flex-direction: column;
        justify-content: center;
        background-color: var(--background-color-normal);
        background-image: radial-gradient(
            circle at center,
            transparent 0,
            transparent 60%,
            var(--background-color-normal) 100%
          ),
          linear-gradient(to right, var(--border-color-lower) 1px, transparent 1px),
          linear-gradient(to bottom, var(--border-color-lower) 1px, transparent 1px);
        background-size:
          auto,
          76px 76px,
          76px 76px;

        & .wrapper {
          padding-top: 192px;
          background-image: url("../assets/images/editor-placeholder.svg");
          background-position: 50% 0;
          background-repeat: no-repeat;
          background-size: 265px 192px;
          color: var(--text-color-normal);
          font-size: var(--text-size-normal);
          text-align: center;
        }
      }

      & .canvas-wrapper {
        height: 0;
        flex-grow: 1;
        padding: var(--spacing-padding-large);
      }
    }
  }
}
</style>
