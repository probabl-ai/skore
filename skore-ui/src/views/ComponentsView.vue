<script setup lang="ts">
import Simplebar from "simplebar-vue";
import type { VisualizationSpec } from "vega-embed";
import { ref } from "vue";

import datatable from "@/assets/fixtures/datatable.json";
import htmlSnippet from "@/assets/fixtures/html-snippet.html?raw";
import markdownString from "@/assets/fixtures/markdown.md?raw";
import multiIndexDatatable from "@/assets/fixtures/multi-index-datatable.json";
import namedIndexDatatable from "@/assets/fixtures/named-index-datatable.json";
import spec from "@/assets/fixtures/vega.json";
import base64Png from "@/assets/images/button-background.png?base64";
import base64Svg from "@/assets/images/editor-placeholder-dark.svg?base64";

import CrossValidationReport from "@/components/CrossValidationReport.vue";
import DataFrameWidget from "@/components/DataFrameWidget.vue";
import DraggableList from "@/components/DraggableList.vue";
import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";
import DynamicContentRasterizer from "@/components/DynamicContentRasterizer.vue";
import EditableList, { type EditableListItemModel } from "@/components/EditableList.vue";
import FloatingTooltip from "@/components/FloatingTooltip.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import SectionHeader from "@/components/SectionHeader.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import TabPanel from "@/components/TabPanel.vue";
import TabPanelContent from "@/components/TabPanelContent.vue";
import TextInput from "@/components/TextInput.vue";
import ToastNotification from "@/components/ToastNotification.vue";
import TreeAccordion, { type TreeAccordionNode } from "@/components/TreeAccordion.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import { generateRandomId } from "@/services/utils";
import { useModalsStore } from "@/stores/modals";
import { useToastsStore } from "@/stores/toasts";

const toastsStore = useToastsStore();
const modalsStore = useModalsStore();

function showToast() {
  toastsStore.addToast(generateRandomId(), "info");
}

function showToastWithCount() {
  toastsStore.addToast(`Info toast`, "info");
}

function showToastWithDuration() {
  toastsStore.addToast(`Toast with duration`, "info", { duration: 5, dismissible: false });
}

const textInputValue = defineModel<string>("value");
const textInputValue2 = defineModel<string>("value2");
const textInputValue3 = defineModel<string>("value3");

function showAlertModal() {
  modalsStore
    .alert(
      "Alert",
      "This is an alert modal with a long text. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ut purus eget sapien. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ut purus eget sapien. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ut purus eget sapien."
    )
    .then(() => {
      console.info("Alert modal closed");
    });
}

function showConfirmModal() {
  modalsStore.confirm("Confirm", "This is a confirm modal").then((result: boolean) => {
    console.info("Confirm modal closed with result:", result);
  });
}

function showPromptModal() {
  modalsStore.prompt("Prompt", "This is a prompt modal", "prompt name").then((result: string) => {
    console.info("Prompt modal closed with result:", result);
  });
}

function onSectionHeaderAction() {
  console.info("Section header action");
}

const fileTreeNodes: TreeAccordionNode[] = [
  {
    name: "fraud",
    enabled: true,
    children: [
      { name: "fraud/accuracy", enabled: true },
      { name: "fraud/accuracy3", enabled: true },
    ],
  },
  {
    name: "fraud2",
    enabled: true,
    children: [
      { name: "fraud2/accuracy", enabled: true },
      { name: "fraud2/accuracy3", enabled: true },
    ],
  },
  {
    name: "nested",
    enabled: true,
    children: [
      {
        name: "nested/fraud2/accuracy",
        enabled: true,
        children: [
          { name: "nested/fraud2/accuracy/self", enabled: true },
          { name: "nested/fraud2/accuracy/self2", enabled: true },
          { name: "nested/fraud2/accuracy/self3", enabled: true },
          { name: "nested/fraud2/accuracy/self", enabled: true },
          { name: "nested/fraud2/accuracy/self2", enabled: true },
          { name: "nested/fraud2/accuracy/self3", enabled: true },
          { name: "nested/fraud2/accuracy/self", enabled: true },
          { name: "nested/fraud2/accuracy/self2", enabled: true },
          { name: "nested/fraud2/accuracy/self3", enabled: true },
          { name: "nested/fraud2/accuracy/self", enabled: true },
          { name: "nested/fraud2/accuracy/self2", enabled: true },
          { name: "nested/fraud2/accuracy/self3", enabled: true },
        ],
      },
      { name: "nested/fraud2/accuracy3", enabled: true, children: [] },
    ],
  },
];

const lastAction = ref<string | null>(null);
const fileTreeItemWithActions: TreeAccordionNode[] = [
  {
    name: "fraud",
    enabled: true,
    children: [
      {
        name: "fraud/accuracy",
        enabled: true,
        actions: [
          { icon: "icon-plus-circle", actionName: "add", enabled: true },
          { icon: "icon-trash", actionName: "delete", enabled: true },
        ],
      },
      {
        name: "fraud/accuracy3",
        enabled: true,
        actions: [
          { icon: "icon-plus-circle", actionName: "add", enabled: true },
          { icon: "icon-trash", actionName: "delete", enabled: true },
        ],
      },
    ],
  },
];

const items = ref<EditableListItemModel[]>([
  { name: "Item 1", icon: "icon-plot", id: generateRandomId() },
  { name: "Item 2", id: generateRandomId() },
  { name: "Item 3", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 4", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 5", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 6", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 7", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 8", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 9", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 10", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 11", icon: "icon-new-document", id: generateRandomId() },
  { name: "Item 12", icon: "icon-new-document", id: generateRandomId() },
]);

function onAddToEditableListAction() {
  items.value.unshift({
    name: "Unnamed",
    icon: "icon-plot",
    isNamed: false,
    id: generateRandomId(),
  });
}

function onEditableListAction(action: string, item: EditableListItemModel) {
  console.info("Add to editable list action", action, item);
  switch (action) {
    case "rename":
      item.isNamed = false;
      break;
    case "duplicate": {
      const index = items.value.indexOf(item) ?? 0;
      items.value.splice(index + 1, 0, {
        name: "Unnamed",
        icon: "icon-plot",
        isNamed: false,
        id: generateRandomId(),
      });
      break;
    }
    case "delete":
      items.value.splice(items.value.indexOf(item), 1);
      break;
  }
}

const lastSelectedItem = ref<string | null>(null);

function addItemToDraggableList(i: number) {
  draggableListData.value.splice(i, 0, {
    name: `${i}`,
    color: `hsl(${(360 / 25) * i}deg, 90%, 50%)`,
    content: Array.from(
      { length: Math.floor(Math.random() * 10) + 1 }, // Random number of items between 1 and 10
      () => String.fromCharCode(97 + Math.floor(Math.random() * 26)) // Random lowercase letter
    ),
  });
}

const draggableListData = ref(
  Array.from({ length: 25 }, (v, i) => ({
    name: `${i}`,
    color: `hsl(${(360 / 25) * i}deg, 90%, 50%)`,
    content: Array.from(
      { length: Math.floor(Math.random() * 10) + 1 }, // Random number of items between 1 and 10
      () => String.fromCharCode(97 + Math.floor(Math.random() * 26)) // Random lowercase letter
    ),
  }))
);

function onDragStart(event: DragEvent) {
  if (event.dataTransfer) {
    event.dataTransfer.setData("application/x-skore-item-name", "drag-me");
  }
}

const currentDropPosition = ref<number>();

function onItemDrop(event: DragEvent) {
  if (event.dataTransfer) {
    if (currentDropPosition.value !== undefined) {
      addItemToDraggableList(currentDropPosition.value);
    }
  }
}

const isCached = ref(false);
</script>

<template>
  <main>
    <h1>Components library</h1>
    <TabPanel>
      <TabPanelContent name="markdown">
        <MarkdownWidget :source="markdownString" />
      </TabPanelContent>
      <TabPanelContent name="vega">
        <VegaWidget :spec="spec as VisualizationSpec" />
      </TabPanelContent>
      <TabPanelContent name="dataframe" class="dataframe">
        <div>Simple index</div>
        <DataFrameWidget
          :index="datatable.index"
          :columns="datatable.columns"
          :data="datatable.data"
          :index-names="datatable.index_names"
        />
        <div>multi index</div>
        <DataFrameWidget
          :index="multiIndexDatatable.index"
          :columns="multiIndexDatatable.columns"
          :data="multiIndexDatatable.data"
          :index-names="multiIndexDatatable.index_names"
        />
        <div>named index</div>
        <DataFrameWidget
          :index="namedIndexDatatable.index"
          :columns="namedIndexDatatable.columns"
          :data="namedIndexDatatable.data"
          :index-names="namedIndexDatatable.index_names"
        />
      </TabPanelContent>
      <TabPanelContent name="images">
        <div class="images">
          <div>
            <p>png</p>
            <ImageWidget mediaType="image/png" :base64-src="base64Png" alt="base64 png" />
          </div>
          <div>
            <p>svg</p>
            <ImageWidget mediaType="image/svg+xml" :base64-src="base64Svg" alt="svg" />
          </div>
        </div>
      </TabPanelContent>
      <TabPanelContent name="buttons">
        <div class="buttons">
          <p>
            <SimpleButton label="hey ho" :is-primary="true" />
            primary button with label
          </p>
          <p>
            <SimpleButton label="hey ho" :is-primary="true" icon="icon-pie-chart" />
            primary button with label and icon
          </p>
          <p>
            <SimpleButton :is-primary="true" icon="icon-pie-chart" />
            primary button with icon
          </p>
          <p>
            <SimpleButton label="hey ho" />
            button with label
          </p>
          <p>
            <SimpleButton label="hey ho" icon="icon-pie-chart" />
            button with label and icon
          </p>
          <p>
            <SimpleButton icon="icon-pie-chart" />
            button with icon
          </p>
          <p>
            <SimpleButton label="hey ho" icon="icon-pie-chart" icon-position="right" />
            button with icon on the right
          </p>
          <hr />
          <p>
            <DropdownButton label="hey ho" icon="icon-pie-chart" :is-primary="true">
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" />
              <DropdownButtonItem icon="icon-pie-chart" />
              <DropdownButtonItem label="hey ho" />
            </DropdownButton>
            primary dropdown button with label and icon
          </p>
          <p>
            <DropdownButton label="hey ho" :is-primary="true">
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" />
              <DropdownButtonItem icon="icon-pie-chart" />
              <DropdownButtonItem label="hey ho" />
            </DropdownButton>
            primary dropdown button with label
          </p>
          <p style="text-align: right">
            <DropdownButton icon="icon-pie-chart" :is-primary="true" align="right">
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" />
              <DropdownButtonItem icon="icon-pie-chart" />
              <DropdownButtonItem label="hey ho" />
            </DropdownButton>
            primary dropdown button with icon
          </p>
          <p>
            <DropdownButton label="hey ho" icon="icon-pie-chart">
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" />
              <DropdownButtonItem icon="icon-pie-chart" />
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" icon-position="right" />
            </DropdownButton>
            dropdown button with label and icon
          </p>
          <p style="text-align: right">
            <DropdownButton label="hey ho" align="right">
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" />
              <DropdownButtonItem icon="icon-pie-chart" />
              <DropdownButtonItem label="hey ho" />
            </DropdownButton>
            dropdown button with label
          </p>
          <p>
            <DropdownButton icon="icon-pie-chart">
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" />
              <DropdownButtonItem icon="icon-pie-chart" />
              <DropdownButtonItem label="hey ho" />
            </DropdownButton>
            dropdown button with icon
          </p>
          <hr />
          <p>
            <SimpleButton label="hey ho" :is-inline="true" />
            inline button
          </p>
          <p>
            <DropdownButton icon="icon-pie-chart" label="button" :is-inline="true">
              <DropdownButtonItem label="hey ho" icon="icon-pie-chart" />
              <DropdownButtonItem icon="icon-pie-chart" />
              <DropdownButtonItem label="hey ho" />
            </DropdownButton>
            inline dropdown button with icon
          </p>
        </div>
      </TabPanelContent>
      <TabPanelContent name="toasts">
        <div class="toasts">
          <ToastNotification message="Info toast" type="info" id="info-toast" />
          <ToastNotification message="Success toast" type="success" id="success-toast" />
          <ToastNotification message="Warning toast" type="warning" id="warning-toast" />
          <ToastNotification message="Error toast" type="error" id="error-toast" />
        </div>
        <SimpleButton label="Show unique toast" @click="showToast" />
        <SimpleButton label="Show toast with count" @click="showToastWithCount" />
        <SimpleButton label="Show toast with duration" @click="showToastWithDuration" />
      </TabPanelContent>
      <TabPanelContent name="inputs">
        <div class="text-inputs">
          <p>
            Regular text input
            <TextInput v-model="textInputValue" /> value: {{ textInputValue }}
          </p>
          <p>
            Text input with icon
            <TextInput v-model="textInputValue2" icon="icon-magnifying-glass" /> value:
            {{ textInputValue2 }}
          </p>
          <p>
            Text input with icon and placeholder
            <TextInput
              v-model="textInputValue3"
              icon="icon-pie-chart"
              placeholder="Enter your name"
            />
            value: {{ textInputValue3 }}
          </p>
        </div>
      </TabPanelContent>
      <TabPanelContent name="alert">
        <SimpleButton label="Show alert modal" @click="showAlertModal" />
        <SimpleButton label="Show confirm modal" @click="showConfirmModal" />
        <SimpleButton label="Show prompt modal" @click="showPromptModal" />
      </TabPanelContent>
      <TabPanelContent name="headers">
        <SectionHeader title="Section header" />
        <SectionHeader
          title="Section header with action"
          action-icon="icon-search"
          @action="onSectionHeaderAction"
        />
        <SectionHeader title="Section header with subtitle" subtitle="Subtitle" />
      </TabPanelContent>
      <TabPanelContent name="accordion">
        <TreeAccordion :nodes="fileTreeNodes" />
        <div style="margin-top: 20px">last item action {{ lastAction }}</div>
        <TreeAccordion
          :nodes="fileTreeItemWithActions"
          @item-action="(action, itemName) => (lastAction = `${action} ${itemName}`)"
        />
      </TabPanelContent>
      <TabPanelContent name="editable list" class="editable-list-tab">
        <div class="header">
          Editable List as 2 way data binding... item list is:
          <ul>
            <li v-for="item in items" :key="item.name">{{ item.name }} (id: {{ item.id }})</li>
          </ul>
          It also emit an event when an item is selected. Last selected item: {{ lastSelectedItem }}
        </div>
        <div class="editable-list-container">
          <SectionHeader
            title="Editable list"
            action-icon="icon-plus-circle"
            @action="onAddToEditableListAction"
          />
          <Simplebar class="editable-list-container-scrollable">
            <EditableList
              v-model:items="items"
              :actions="[
                { label: 'rename', emitPayload: 'rename', icon: 'icon-edit' },
                { label: 'duplicate', emitPayload: 'duplicate', icon: 'icon-copy' },
                { label: 'delete', emitPayload: 'delete', icon: 'icon-trash' },
              ]"
              @action="onEditableListAction"
              @select="lastSelectedItem = $event"
            />
          </Simplebar>
        </div>
      </TabPanelContent>
      <TabPanelContent name="icons">
        <div class="icons">
          <div>icon-bar-chart <i class="icon icon-bar-chart"></i></div>
          <div>icon-branch <i class="icon icon-branch"></i></div>
          <div>icon-calendar <i class="icon icon-calendar"></i></div>
          <div>icon-check <i class="icon icon-check"></i></div>
          <div>icon-chevron-down <i class="icon icon-chevron-down"></i></div>
          <div>icon-chevron-left <i class="icon icon-chevron-left"></i></div>
          <div>icon-chevron-right <i class="icon icon-chevron-right"></i></div>
          <div>icon-chevron-up <i class="icon icon-chevron-up"></i></div>
          <div>icon-copy <i class="icon icon-copy"></i></div>
          <div>icon-dashboard <i class="icon icon-dashboard"></i></div>
          <div>icon-edit <i class="icon icon-edit"></i></div>
          <div>icon-error-circle <i class="icon icon-error-circle"></i></div>
          <div>icon-folder <i class="icon icon-folder"></i></div>
          <div>icon-gift <i class="icon icon-gift"></i></div>
          <div>icon-handle <i class="icon icon-handle"></i></div>
          <div>icon-hard-drive <i class="icon icon-hard-drive"></i></div>
          <div>icon-history <i class="icon icon-history"></i></div>
          <div>icon-info-circle <i class="icon icon-info-circle"></i></div>
          <div>icon-large-bar-chart <i class="icon icon-large-bar-chart"></i></div>
          <div>icon-left-double-chevron <i class="icon icon-left-double-chevron"></i></div>
          <div>icon-list-sparkle <i class="icon icon-list-sparkle"></i></div>
          <div>icon-maximize <i class="icon icon-maximize"></i></div>
          <div>icon-more <i class="icon icon-more"></i></div>
          <div>icon-new-document <i class="icon icon-new-document"></i></div>
          <div>icon-pie-chart <i class="icon icon-pie-chart"></i></div>
          <div>icon-pill <i class="icon icon-pill"></i></div>
          <div>icon-playground <i class="icon icon-playground"></i></div>
          <div>icon-plot <i class="icon icon-plot"></i></div>
          <div>icon-plus-circle <i class="icon icon-plus-circle"></i></div>
          <div>icon-plus <i class="icon icon-plus"></i></div>
          <div>icon-podium <i class="icon icon-podium"></i></div>
          <div>icon-recent-document <i class="icon icon-recent-document"></i></div>
          <div>icon-search <i class="icon icon-search"></i></div>
          <div>icon-success-circle <i class="icon icon-success-circle"></i></div>
          <div>icon-text <i class="icon icon-text"></i></div>
          <div>icon-trash <i class="icon icon-trash"></i></div>
          <div>icon-warning-circle <i class="icon icon-warning-circle"></i></div>
          <div>icon-warning <i class="icon icon-warning"></i></div>
          <div>icon-square-cursor <i class="icon icon-square-cursor"></i></div>
        </div>
      </TabPanelContent>
      <TabPanelContent name="draggable">
        <div>Item order: {{ draggableListData.map((item) => item.name).join(", ") }}</div>
        <div>Drop position: {{ currentDropPosition }}</div>
        <div>
          <SimpleButton
            label="add item"
            :is-primary="true"
            @click="addItemToDraggableList(draggableListData.length + 1)"
          />
          <div class="drag-me" draggable="true" @dragstart="onDragStart($event)">
            drag me to the list
          </div>
        </div>
        <Simplebar class="draggable-list-container">
          <DraggableList
            v-model:items="draggableListData"
            auto-scroll-container-selector=".draggable-list-container"
            v-model:current-drop-position="currentDropPosition"
            @drop="onItemDrop($event)"
            @dragover.prevent
          >
            <template #item="{ name: id, color, content }">
              <div :style="{ backgroundColor: color, color: 'white' }">
                <span>ID: {{ id }}</span>
                <ul>
                  <li v-for="(c, i) in content" :key="i">{{ c }}</li>
                </ul>
              </div>
            </template>
          </DraggableList>
        </Simplebar>
      </TabPanelContent>
      <TabPanelContent name="cached">
        <label>
          Cache the following widget
          <input type="checkbox" v-model="isCached" />
        </label>
        <DynamicContentRasterizer :isRasterized="isCached">
          <div>lorem ipsum dolor sit amet</div>
          <HtmlSnippetWidget :src="htmlSnippet" />
        </DynamicContentRasterizer>
      </TabPanelContent>
      <TabPanelContent name="tooltips" class="floating-tooltip-tab">
        <div>
          <FloatingTooltip text="Tooltip on div">bottom tooltip</FloatingTooltip>
        </div>
        <div>
          <FloatingTooltip text="Tooltip on div" placement="top">top tooltip</FloatingTooltip>
        </div>
        <div>
          <FloatingTooltip text="Tooltip on div" placement="left">left tooltip</FloatingTooltip>
        </div>
        <div>
          <FloatingTooltip text="Tooltip on div" placement="right">right tooltip</FloatingTooltip>
        </div>
        <div>
          <FloatingTooltip placement="right">
            html tooltip
            <template #tooltipContent>
              <span style="color: red">red content</span>
            </template>
          </FloatingTooltip>
        </div>
        <div>
          <FloatingTooltip text="Tooltip on button">
            <SimpleButton label="button with tooltip" />
          </FloatingTooltip>
        </div>
        <div>
          <FloatingTooltip text="Tooltip on button bottom-start" placement="bottom-start">
            bottom start
          </FloatingTooltip>
        </div>
        <div>
          <FloatingTooltip text="Tooltip on button bottom-end" placement="bottom-end">
            bottom end
          </FloatingTooltip>
        </div>
      </TabPanelContent>
      <TabPanelContent name="cross val" class="cross-val">
        <CrossValidationReport />
      </TabPanelContent>
    </TabPanel>
  </main>
</template>

<style scoped>
main {
  padding: 0 5vw;

  & h1 {
    margin: var(--spacing-8) 0;
  }
}

.dataframe {
  margin-top: 10px;
}

.images {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  place-items: center center;
}

.buttons {
  & > p {
    padding: 10px;
  }
}

.toasts {
  padding-top: 10px;

  & > * {
    margin-bottom: 10px;
  }
}

.text-inputs {
  & > p {
    padding: 10px;
  }
}

.editable-list-tab {
  display: grid;
  padding-top: 10px;
  grid-template-columns: 1fr 1fr;
}

.editable-list-container {
  display: block;
  width: 33vw;
  height: 450px;

  & .editable-list-container-scrollable {
    height: 100%;
    overflow-y: auto;
  }
}

.icons {
  display: grid;
  padding-top: 10px;
  gap: 20px;
  grid-template-columns: 1fr 1fr 1fr 1fr;

  & > div {
    display: flex;
    gap: 10px;
  }

  & [class^="icon-"],
  & [class*=" icon-"] {
    color: var(--color-icon-primary);
    font-size: 20px;
  }
}

.draggable-list-container {
  max-height: 80vh;
  margin-top: 10px;
}

.drag-me {
  width: 200px;
  padding: 10px;
  margin-top: 10px;
  background-color: var(--color-background-secondary);
  cursor: move;
  user-select: none;
}

.floating-tooltip-tab {
  display: grid;
  padding: 40px;
  gap: 40px;
  grid-template-columns: 1fr 1fr 1fr;
}

.cross-val {
  padding: var(--spacing-8);
}
</style>
