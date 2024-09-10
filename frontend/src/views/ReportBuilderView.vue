<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { onBeforeUnmount, onMounted, ref } from "vue";

import DataStoreCanvas from "@/components/DataStoreCanvas.vue";
import DataStoreKeyList from "@/components/DataStoreKeyList.vue";
import SectionHeader from "@/components/SectionHeader.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import { useReportStore } from "@/stores/report";

const reportStore = useReportStore();
const isDropIndicatorVisible = ref(false);
const editor = ref<HTMLDivElement>();
const isInFocusMode = ref(false);

async function onShareReport() {
  /*
  const uri = reportsStore.report?.uri;
  if (uri) {
    const shareable = await fetchShareableBlob(uri);
    if (shareable) {
      saveBlob(
        shareable,
        uri.replace(/\//g, (_, i) => (i === 0 ? "" : "-"))
      );
    }
  }
  */
  alert("Not implemented yet");
}

function onFocusMode() {
  isInFocusMode.value = !isInFocusMode.value;
}

function onItemDrop(event: DragEvent) {
  isDropIndicatorVisible.value = false;
  if (event.dataTransfer) {
    const key = event.dataTransfer.getData("key");
    reportStore.displayKey(key);
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

onMounted(() => {
  reportStore.startBackendPolling();
});

onBeforeUnmount(() => reportStore.stopBackendPolling());
</script>

<template>
  <main>
    <article>
      <div class="items" v-if="reportStore.report && !isInFocusMode">
        <SectionHeader title="Items" icon="icon-pie-chart" />
        <Simplebar class="key-list">
          <DataStoreKeyList title="Info" icon="icon-text" :keys="Object.keys(reportStore.report)" />
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
          <SimpleButton icon="icon-focus" @click="onFocusMode" />
          <h1>Report</h1>
          <SimpleButton label="Share report" @click="onShareReport" :is-primary="true" />
        </div>
        <div class="drop-indicator" :class="{ visible: isDropIndicatorVisible }"></div>
        <Transition name="fade">
          <div
            v-if="!isDropIndicatorVisible && reportStore.layout.length === 0"
            class="placeholder"
          >
            <div class="wrapper">No item selected yet, start by selecting a Skore.</div>
          </div>

          <Simplebar class="canvas-wrapper" v-else ref="reportScrollBar">
            <DataStoreCanvas />
          </Simplebar>
        </Transition>
      </div>
    </article>
    <div class="not-found" v-if="reportStore.report === null">
      <div class="not-found-header">Empty workspace.</div>
      <p>No Skore has been created, this worskpace is empty.</p>
    </div>
  </main>
</template>

<style scoped>
@media (prefers-color-scheme: dark) {
  main {
    --sad-face-image: url("../assets/images/sad-face-dark.svg");
    --not-found-image: url("../assets/images/not-found-dark.png");
    --editor-placeholder-image: url("../assets/images/editor-placeholder-dark.svg");
  }
}

@media (prefers-color-scheme: light) {
  main {
    --sad-face-image: url("../assets/images/sad-face-light.svg");
    --not-found-image: url("../assets/images/not-found-light.png");
    --editor-placeholder-image: url("../assets/images/editor-placeholder-light.svg");
  }
}

main {
  display: flex;
  flex-direction: row;

  nav,
  article {
    height: 100dvh;
  }

  nav {
    display: flex;
    overflow: hidden;
    width: 240px;
    min-width: 0;
    flex-direction: column;
    flex-shrink: 0;
    border-right: solid 1px var(--border-color-normal);

    & .file-trees,
    & .empty-tree {
      height: 0;
      flex-grow: 1;
    }

    & .empty-tree {
      display: flex;
      flex-direction: column;
      justify-content: center;
      background-image: var(--sad-face-image);
      background-position: 50% calc(50% - 24px);
      background-repeat: no-repeat;
      background-size: 24px;
      color: var(--text-color-normal);
      font-size: var(--text-size-small);
      text-align: center;
    }
  }

  article,
  .not-found {
    display: flex;
    overflow: hidden;
    min-width: 0;
    flex-grow: 1;
  }

  article {
    flex-direction: row;

    & .items {
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
      overflow: hidden;
      min-width: 0;
      max-height: 100vh;
      flex: auto;
      flex-direction: column;

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
          background-image: var(--editor-placeholder-image);
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

  .not-found {
    flex-direction: column;
    justify-content: center;
    background-image: var(--not-found-image);
    background-position: 50% calc(50% - 62px);
    background-repeat: no-repeat;
    background-size: 109px 82px;
    color: var(--text-color-normal);
    font-size: var(--text-size-small);
    text-align: center;

    & .not-found-header {
      margin-bottom: var(--spacing-padding-small);
      color: var(--text-color-highlight);
      font-size: var(--text-size-large);
    }
  }
}
</style>
