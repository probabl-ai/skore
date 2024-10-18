<script setup lang="ts">
import { format, formatDistance } from "date-fns";
import Simplebar from "simplebar-vue";
import { onBeforeUnmount, onMounted, ref } from "vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import DraggableList from "@/components/DraggableList.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import ProjectViewCard from "@/components/ProjectViewCard.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import { fetchShareableBlob } from "@/services/api";
import { saveBlob } from "@/services/utils";
import { useProjectStore } from "@/stores/project";
import { useToastsStore } from "@/stores/toasts";
import ProjectElementList from "@/views/project/ProjectElementList.vue";
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

async function onShareView() {
  const currentView = projectStore.currentView;
  if (currentView) {
    const shareable = await fetchShareableBlob(currentView);
    if (shareable) {
      const formattedDate = format(new Date(), "yyyy-MM-dd-HH-mm");
      saveBlob(shareable, `${formattedDate}-${currentView}.html`);
    }
  }
}

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
      <ProjectElementList />
    </div>
    <div ref="editor" class="editor">
      <div class="editor-header">
        <SimpleButton icon="icon-maximize" @click="onFocusMode" />
        <h1>{{ projectStore.currentView }}</h1>
        <SimpleButton label="Share view" @click="onShareView" :is-primary="true" />
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
          <div class="wrapper" v-else>
            No elements in this view, start by dragging or double clicking an element from the tree
            on the left.
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
  <main class="not-found" v-else>
    <div class="not-found-header">Empty workspace.</div>
    <p>No Skore has been created, this worskpace is empty.</p>
  </main>
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
          font-weight: var(--text-weight-title);
          text-align: center;
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

  .not-found {
    height: 100vh;
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
