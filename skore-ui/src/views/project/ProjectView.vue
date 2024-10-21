<script setup lang="ts">
import { formatDistance } from "date-fns";
import Simplebar from "simplebar-vue";
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import DraggableList from "@/components/DraggableList.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import ProjectViewCard from "@/components/ProjectViewCard.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import { useProjectStore } from "@/stores/project";
import { useToastsStore } from "@/stores/toasts";
import ProjectItemList from "@/views/project/ProjectItemList.vue";
import ProjectViewList from "@/views/project/ProjectViewList.vue";

const props = defineProps({
  showCardActions: {
    type: Boolean,
    default: true,
  },
});

const projectStore = useProjectStore();
const isInFocusMode = ref(false);
const currentDropPosition = ref<number>();
const toastsStore = useToastsStore();

function onFocusMode() {
  isInFocusMode.value = !isInFocusMode.value;
}

function onCardRemoved(key: string) {
  if (projectStore.currentView) {
    projectStore.hideKey(projectStore.currentView, key);
  }
}

function onItemDrop(event: DragEvent) {
  if (projectStore.currentView) {
    if (event.dataTransfer) {
      const itemName = event.dataTransfer.getData("application/x-skore-item-name");
      projectStore.displayKey(projectStore.currentView, itemName, currentDropPosition.value ?? 0);
    }
  } else {
    toastsStore.addToast("No view selected", "error");
  }
}

function getItemSubtitle(created_at: Date, updated_at: Date) {
  const now = new Date();
  return `Created ${formatDistance(created_at, now)} ago, updated ${formatDistance(updated_at, now)} ago`;
}

watch(
  () => projectStore.currentViewItems,
  async (newItems, oldItems) => {
    // when the item list is updated ensure that scroll let the latest added item be visible
    if (newItems.length > oldItems.length) {
      nextTick(() => {
        const lastItemElement = document.querySelector(".editor-container .item:last-child");
        if (lastItemElement) {
          lastItemElement.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    }
  }
);

onMounted(async () => {
  await projectStore.startBackendPolling();
});

onBeforeUnmount(() => {
  projectStore.stopBackendPolling();
});
</script>

<template>
  <main class="project-view" v-if="projectStore.items !== null">
    <div class="left-panel" v-if="projectStore.items && !isInFocusMode">
      <ProjectViewList />
      <ProjectItemList />
    </div>
    <div ref="editor" class="editor">
      <div class="editor-header">
        <SimpleButton icon="icon-maximize" @click="onFocusMode" />
        <h1>{{ projectStore.currentView }}</h1>
      </div>
      <Transition name="fade">
        <div
          v-if="
            projectStore.currentView === null ||
            projectStore.views[projectStore.currentView] === undefined ||
            projectStore.views[projectStore.currentView]?.length === 0
          "
          class="placeholder"
          @drop="onItemDrop($event)"
          @dragover.prevent
        >
          <div class="wrapper" v-if="projectStore.currentView === null">No view selected.</div>
          <div class="dropzone" v-else>
            <div class="wrapper">The view is empty, start by dropping an element.</div>
          </div>
        </div>
        <Simplebar class="editor-container" v-else>
          <DraggableList
            v-model:items="projectStore.currentViewItems"
            auto-scroll-container-selector=".editor-container"
            v-model:current-drop-position="currentDropPosition"
            @drop="onItemDrop($event)"
            @dragover.prevent
          >
            <template #item="{ key, mediaType, data, createdAt, updatedAt }">
              <ProjectViewCard
                :key="key"
                :title="key.toString()"
                :subtitle="getItemSubtitle(createdAt, updatedAt)"
                :showActions="props.showCardActions"
                @card-removed="onCardRemoved(key)"
              >
                <DataFrameWidget
                  v-if="mediaType === 'application/vnd.dataframe+json'"
                  :columns="data.columns"
                  :data="data.data"
                  :index="data.index"
                  :index-names="data.index_names"
                />
                <ImageWidget
                  v-if="
                    ['image/svg+xml', 'image/png', 'image/jpeg', 'image/webp'].includes(mediaType)
                  "
                  :mediaType="mediaType"
                  :base64-src="data"
                  :alt="key.toString()"
                />
                <MarkdownWidget v-if="mediaType === 'text/markdown'" :source="data" />
                <VegaWidget v-if="mediaType === 'application/vnd.vega.v5+json'" :spec="data" />
                <PlotlyWidget v-if="mediaType === 'application/vnd.plotly.v1+json'" :spec="data" />
                <HtmlSnippetWidget
                  v-if="mediaType === 'application/vnd.sklearn.estimator+html'"
                  :src="data"
                />
                <HtmlSnippetWidget v-if="mediaType === 'text/html'" :src="data" />
              </ProjectViewCard>
            </template>
          </DraggableList>
        </Simplebar>
      </Transition>
    </div>
  </main>
  <main class="not-found" v-else>No Skore has been created, this worskpace is empty.</main>
</template>

<style scoped>
@media (prefers-color-scheme: dark) {
  main {
    --sad-face-image: url("../../assets/images/sad-face-dark.svg");
    --not-found-image: url("../../assets/images/not-found-dark.png");
    --editor-placeholder-image: url("../../assets/images/editor-placeholder-dark.svg");
  }
}

@media (prefers-color-scheme: light) {
  main {
    --sad-face-image: url("../../assets/images/sad-face-light.svg");
    --not-found-image: url("../../assets/images/not-found-light.png");
    --editor-placeholder-image: url("../../assets/images/editor-placeholder-light.svg");
  }
}

main {
  display: flex;
  overflow: hidden;
  width: 100vw;
  min-width: 0;
  height: 100vh;
  flex-direction: row;

  &.project-view {
    height: 100dvh;
    flex-direction: row;

    & .left-panel {
      display: flex;
      width: 292px;
      flex-direction: column;
      flex-shrink: 0;
      border-right: solid 1px var(--border-color-normal);

      & .views-list {
        z-index: 2;
      }

      & .keys-list {
        z-index: 1;
        height: 0;
        flex: 1;
      }
    }

    & .editor {
      --editor-height: 44px;

      display: flex;
      min-width: 0;
      max-height: 100vh;
      flex: auto;
      flex-direction: column;

      & .editor-header {
        display: flex;
        height: var(--editor-height);
        align-items: center;
        padding: var(--spacing-padding-large);
        border-right: solid var(--border-width-normal) var(--border-color-normal);
        border-bottom: solid var(--border-width-normal) var(--border-color-normal);
        background-color: var(--background-color-elevated);

        & h1 {
          flex-grow: 1;
          color: var(--text-color-normal);
          font-size: var(--text-size-title);
          font-weight: var(--text-weight-highlight);
          text-align: center;
        }
      }

      & .placeholder {
        display: flex;
        height: 100%;
        flex-direction: column;
        justify-content: center;
        background-color: var(--background-color-normal);

        &:not(:has(.dropzone)) {
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
        }

        & .wrapper {
          padding-top: 225px;
          margin: var(--spacing-padding-large);
          background-image: var(--editor-placeholder-image);
          background-position: 50% 0;
          background-repeat: no-repeat;
          background-size: 265px 192px;
          color: var(--text-color-normal);
          font-size: var(--text-size-normal);
          text-align: center;
        }

        & .dropzone {
          display: flex;
          height: 100%;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          border-radius: 17px;
          margin: var(--spacing-padding-large);
          background: var(--background-color-normal);
          background-color: var(--background-color-elevated);
          background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='17' ry='17' stroke='%23BABBBDFF' stroke-width='1' stroke-dasharray='11%2c11' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
        }
      }

      & .editor-container {
        height: 0;
        flex: 1;
        padding: var(--spacing-padding-large);

        & .draggable {
          min-height: calc(100dvh - var(--editor-height) - var(--spacing-padding-large) * 2);
          gap: var(--spacing-gap-large);
        }
      }
    }
  }

  &.not-found {
    display: flex;
    height: 100vh;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-color-normal);
    font-size: var(--text-size-normal);
  }
}
</style>
