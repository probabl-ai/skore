<script setup lang="ts">
import { format } from "date-fns";
import Simplebar from "simplebar-vue";
import { onBeforeUnmount, onMounted, ref } from "vue";

import EditableList, { type EditableListItemModel } from "@/components/EditableList.vue";
import ProjectViewCanvas from "@/components/ProjectViewCanvas.vue";
import SectionHeader from "@/components/SectionHeader.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import TreeAccordion from "@/components/TreeAccordion.vue";
import { fetchShareableBlob } from "@/services/api";
import { generateRandomId, saveBlob } from "@/services/utils";
import { useProjectStore } from "@/stores/project";

const projectStore = useProjectStore();
const isDropIndicatorVisible = ref(false);
const editor = ref<HTMLDivElement>();
const isInFocusMode = ref(false);
const views = ref<EditableListItemModel[]>([]);
let unsavedViewsIds: string[] = [];

async function onShareReport() {
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

function onItemDrop(event: DragEvent) {
  isDropIndicatorVisible.value = false;
  if (event.dataTransfer) {
    const key = event.dataTransfer.getData("key");
    if (projectStore.currentView) {
      projectStore.displayKey(projectStore.currentView, key);
    }
  }
}

function onDragEnter() {
  if (projectStore.currentView) {
    isDropIndicatorVisible.value = true;
  }
}

function onDragLeave(event: DragEvent) {
  if (event.target == editor.value) {
    isDropIndicatorVisible.value = false;
  }
}

function onItemSelected(key: string) {
  if (projectStore.currentView) {
    projectStore.displayKey(projectStore.currentView, key);
  }
}

function onViewSelected(view: string) {
  projectStore.currentView = view;
}

async function onViewRenamed(oldName: string, newName: string, item: EditableListItemModel) {
  // user can rename an existing view
  // and rename a new view that is not known by the backend
  if (unsavedViewsIds.includes(item.id)) {
    await projectStore.createView(newName);
    unsavedViewsIds = unsavedViewsIds.filter((id) => id !== item.id);
  } else {
    await projectStore.renameView(oldName, newName);
  }
  // auto switch the view to the renamed one
  onViewSelected(newName);
}

async function onViewsListAction(action: string, item: EditableListItemModel) {
  switch (action) {
    case "rename": {
      item.isUnnamed = true;
      break;
    }
    case "duplicate": {
      const index = views.value.indexOf(item) ?? 0;
      const newName = `${item.name} - copy`;
      views.value.splice(index + 1, 0, {
        name: newName,
        icon: "icon-new-document",
        isUnnamed: true,
        id: generateRandomId(),
      });
      unsavedViewsIds.push(newName);
      await projectStore.duplicateView(item.name, newName);
      break;
    }
    case "delete": {
      views.value.splice(views.value.indexOf(item), 1);
      await projectStore.deleteView(item.name);
      break;
    }
  }
}

function onAddView() {
  const id = generateRandomId();
  views.value.push({ name: "New view", icon: "icon-new-document", isUnnamed: true, id });
  unsavedViewsIds.push(id);
}

onMounted(async () => {
  await projectStore.startBackendPolling();
  views.value = Object.keys(projectStore.views).map((key) => ({
    name: key,
    icon: "icon-new-document",
    isUnnamed: false,
    id: generateRandomId(),
  }));
});
onBeforeUnmount(() => {
  projectStore.stopBackendPolling();
});
</script>

<template>
  <main>
    <article v-if="projectStore.items !== null">
      <div class="items" v-if="projectStore.items && !isInFocusMode">
        <SectionHeader
          title="Views"
          icon="icon-recent-document"
          action-icon="icon-plus-circle"
          @action="onAddView"
        />
        <Simplebar class="views-list">
          <EditableList
            v-model:items="views"
            :actions="[
              { label: 'rename', emitPayload: 'rename', icon: 'icon-edit' },
              { label: 'duplicate', emitPayload: 'duplicate', icon: 'icon-copy' },
              { label: 'delete', emitPayload: 'delete', icon: 'icon-trash' },
            ]"
            @action="onViewsListAction"
            @select="onViewSelected"
            @rename="onViewRenamed"
          />
        </Simplebar>
        <SectionHeader title="Elements" icon="icon-pie-chart" />
        <Simplebar class="key-list">
          <TreeAccordion :nodes="projectStore.keysAsTree()" @item-selected="onItemSelected" />
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
          <SimpleButton icon="icon-maximize" @click="onFocusMode" />
          <h1>{{ projectStore.currentView }}</h1>
          <SimpleButton label="Share report" @click="onShareReport" :is-primary="true" />
        </div>
        <div class="drop-indicator" :class="{ visible: isDropIndicatorVisible }"></div>
        <Transition name="fade">
          <div
            v-if="
              projectStore.currentView === null ||
              projectStore.views[projectStore.currentView] === undefined ||
              projectStore.views[projectStore.currentView]?.length === 0
            "
            class="placeholder"
          >
            <div class="wrapper" v-if="projectStore.currentView === null">No view selected.</div>
            <div class="wrapper" v-else>
              No elements in this view, start by dragging or double clicking an element from the
              tree on the left.
            </div>
          </div>
          <Simplebar class="canvas-wrapper" v-else>
            <ProjectViewCanvas />
          </Simplebar>
        </Transition>
      </div>
    </article>
    <div class="not-found" v-else>
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

  article,
  .not-found {
    display: flex;
    overflow: hidden;
    min-width: 0;
    flex-grow: 1;
  }

  article {
    height: 100dvh;
    flex-direction: row;

    & .items {
      display: flex;
      width: 292px;
      flex-direction: column;
      flex-shrink: 0;
      border-right: solid 1px var(--border-color-normal);

      & .views-list {
        z-index: 2;
        max-height: 20vh;
        background-color: var(--background-color-elevated-high);
      }

      & .key-list {
        z-index: 1;
        height: 0;
        flex-grow: 1;
        padding: var(--spacing-padding-large);
        background-color: var(--background-color-elevated-high);
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
