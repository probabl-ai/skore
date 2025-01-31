<script setup lang="ts">
import Simplebar from "simplebar-vue";
import type { VisualizationSpec } from "vega-embed";
import { ref, useTemplateRef } from "vue";

import datatable from "@/assets/fixtures/datatable.json";
import htmlSnippet from "@/assets/fixtures/html-snippet.html?raw";
import markdownString from "@/assets/fixtures/markdown.md?raw";
import multiIndexDatatable from "@/assets/fixtures/multi-index-datatable.json";
import namedIndexDatatable from "@/assets/fixtures/named-index-datatable.json";
import spec from "@/assets/fixtures/vega.json";
import base64Png from "@/assets/images/button-background.png?base64";
import base64Svg from "@/assets/images/editor-placeholder-dark.svg?base64";

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
import RichTextEditor from "@/components/RichTextEditor.vue";
import SectionHeader from "@/components/SectionHeader.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import SlideToggle from "@/components/SlideToggle.vue";
import TabPanel from "@/components/TabPanel.vue";
import TabPanelContent from "@/components/TabPanelContent.vue";
import TextInput from "@/components/TextInput.vue";
import ToastNotification from "@/components/ToastNotification.vue";
import TreeAccordion, { type TreeAccordionNode } from "@/components/TreeAccordion.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import type { DetailSectionDto, PlotDto, PrimaryResultsDto } from "@/dto";
import { generateRandomId } from "@/services/utils";
import { useModalsStore } from "@/stores/modals";
import { useThemesStore } from "@/stores/themes";
import { useToastsStore } from "@/stores/toasts";

const toastsStore = useToastsStore();
const modalsStore = useModalsStore();
const themesStore = useThemesStore();

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

const results: PrimaryResultsDto = {
  scalarResults: [
    { name: "toto", value: 4.32, favorability: "greater_is_better" },
    { name: "tata", value: 4.32, favorability: "greater_is_better" },
    { name: "titi", value: 4.32, stddev: 1, favorability: "greater_is_better" },
    { name: "tutu", value: 4.32, favorability: "greater_is_better" },
    { name: "stab", value: 0.4, label: "Good", favorability: "greater_is_better" },
    { name: "titi", value: 4.32, stddev: 1, favorability: "greater_is_better" },
    { name: "tutu", value: 4.32, favorability: "greater_is_better" },
    {
      name: "stab",
      value: 0.9,
      label: "Good",
      description: "your blabla is good",
      favorability: "greater_is_better",
    },
  ],
  tabularResults: [
    {
      name: "abracadaraabracadaraabracadaraabracadara",
      columns: Array.from({ length: 50 }, (_, i) => i),
      data: [
        Array.from({ length: 50 }, () => Math.random().toFixed(4)),
        Array.from({ length: 50 }, () => Math.random().toFixed(4)),
        Array.from({ length: 50 }, () => Math.random().toFixed(4)),
        Array.from({ length: 50 }, () => Math.random().toFixed(4)),
        Array.from({ length: 50 }, () => Math.random().toFixed(4)),
        Array.from({ length: 50 }, () => Math.random().toFixed(4)),
      ],
      favorability: Array.from({ length: 50 }, (_, i) =>
        i % 2 === 0 ? "greater_is_better" : "lower_is_better"
      ),
    },
    {
      name: "b",
      columns: ["Accuracy", "Precision", "Recall", "F1 Score"],
      data: [
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
      ],
      favorability: ["greater_is_better", "greater_is_better", "lower_is_better"],
    },
  ],
};

const fakePlot = {
  data: [{ x: [1, 2, 3], y: [1, 3, 2], type: "bar" }],
  layout: {
    title: { text: "A Figure Specified by a Dictionary" },
    template: {
      data: {
        histogram2dcontour: [
          {
            type: "histogram2dcontour",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        choropleth: [{ type: "choropleth", colorbar: { outlinewidth: 0, ticks: "" } }],
        histogram2d: [
          {
            type: "histogram2d",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        heatmap: [
          {
            type: "heatmap",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        heatmapgl: [
          {
            type: "heatmapgl",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        contourcarpet: [{ type: "contourcarpet", colorbar: { outlinewidth: 0, ticks: "" } }],
        contour: [
          {
            type: "contour",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        surface: [
          {
            type: "surface",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        mesh3d: [{ type: "mesh3d", colorbar: { outlinewidth: 0, ticks: "" } }],
        scatter: [
          { fillpattern: { fillmode: "overlay", size: 10, solidity: 0.2 }, type: "scatter" },
        ],
        parcoords: [{ type: "parcoords", line: { colorbar: { outlinewidth: 0, ticks: "" } } }],
        scatterpolargl: [
          { type: "scatterpolargl", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        bar: [
          {
            error_x: { color: "#2a3f5f" },
            error_y: { color: "#2a3f5f" },
            marker: {
              line: { color: "#E5ECF6", width: 0.5 },
              pattern: { fillmode: "overlay", size: 10, solidity: 0.2 },
            },
            type: "bar",
          },
        ],
        scattergeo: [{ type: "scattergeo", marker: { colorbar: { outlinewidth: 0, ticks: "" } } }],
        scatterpolar: [
          { type: "scatterpolar", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        histogram: [
          {
            marker: { pattern: { fillmode: "overlay", size: 10, solidity: 0.2 } },
            type: "histogram",
          },
        ],
        scattergl: [{ type: "scattergl", marker: { colorbar: { outlinewidth: 0, ticks: "" } } }],
        scatter3d: [
          {
            type: "scatter3d",
            line: { colorbar: { outlinewidth: 0, ticks: "" } },
            marker: { colorbar: { outlinewidth: 0, ticks: "" } },
          },
        ],
        scattermapbox: [
          { type: "scattermapbox", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        scatterternary: [
          { type: "scatterternary", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        scattercarpet: [
          { type: "scattercarpet", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        carpet: [
          {
            aaxis: {
              endlinecolor: "#2a3f5f",
              gridcolor: "white",
              linecolor: "white",
              minorgridcolor: "white",
              startlinecolor: "#2a3f5f",
            },
            baxis: {
              endlinecolor: "#2a3f5f",
              gridcolor: "white",
              linecolor: "white",
              minorgridcolor: "white",
              startlinecolor: "#2a3f5f",
            },
            type: "carpet",
          },
        ],
        table: [
          {
            cells: { fill: { color: "#EBF0F8" }, line: { color: "white" } },
            header: { fill: { color: "#C8D4E3" }, line: { color: "white" } },
            type: "table",
          },
        ],
        barpolar: [
          {
            marker: {
              line: { color: "#E5ECF6", width: 0.5 },
              pattern: { fillmode: "overlay", size: 10, solidity: 0.2 },
            },
            type: "barpolar",
          },
        ],
        pie: [{ automargin: true, type: "pie" }],
      },
      layout: {
        autotypenumbers: "strict",
        colorway: [
          "#636efa",
          "#EF553B",
          "#00cc96",
          "#ab63fa",
          "#FFA15A",
          "#19d3f3",
          "#FF6692",
          "#B6E880",
          "#FF97FF",
          "#FECB52",
        ],
        font: { color: "#2a3f5f" },
        hovermode: "closest",
        hoverlabel: { align: "left" },
        paper_bgcolor: "white",
        plot_bgcolor: "#E5ECF6",
        polar: {
          bgcolor: "#E5ECF6",
          angularaxis: { gridcolor: "white", linecolor: "white", ticks: "" },
          radialaxis: { gridcolor: "white", linecolor: "white", ticks: "" },
        },
        ternary: {
          bgcolor: "#E5ECF6",
          aaxis: { gridcolor: "white", linecolor: "white", ticks: "" },
          baxis: { gridcolor: "white", linecolor: "white", ticks: "" },
          caxis: { gridcolor: "white", linecolor: "white", ticks: "" },
        },
        coloraxis: { colorbar: { outlinewidth: 0, ticks: "" } },
        colorscale: {
          sequential: [
            [0.0, "#0d0887"],
            [0.1111111111111111, "#46039f"],
            [0.2222222222222222, "#7201a8"],
            [0.3333333333333333, "#9c179e"],
            [0.4444444444444444, "#bd3786"],
            [0.5555555555555556, "#d8576b"],
            [0.6666666666666666, "#ed7953"],
            [0.7777777777777778, "#fb9f3a"],
            [0.8888888888888888, "#fdca26"],
            [1.0, "#f0f921"],
          ],
          sequentialminus: [
            [0.0, "#0d0887"],
            [0.1111111111111111, "#46039f"],
            [0.2222222222222222, "#7201a8"],
            [0.3333333333333333, "#9c179e"],
            [0.4444444444444444, "#bd3786"],
            [0.5555555555555556, "#d8576b"],
            [0.6666666666666666, "#ed7953"],
            [0.7777777777777778, "#fb9f3a"],
            [0.8888888888888888, "#fdca26"],
            [1.0, "#f0f921"],
          ],
          diverging: [
            [0, "#8e0152"],
            [0.1, "#c51b7d"],
            [0.2, "#de77ae"],
            [0.3, "#f1b6da"],
            [0.4, "#fde0ef"],
            [0.5, "#f7f7f7"],
            [0.6, "#e6f5d0"],
            [0.7, "#b8e186"],
            [0.8, "#7fbc41"],
            [0.9, "#4d9221"],
            [1, "#276419"],
          ],
        },
        xaxis: {
          gridcolor: "white",
          linecolor: "white",
          ticks: "",
          title: { standoff: 15 },
          zerolinecolor: "white",
          automargin: true,
          zerolinewidth: 2,
        },
        yaxis: {
          gridcolor: "white",
          linecolor: "white",
          ticks: "",
          title: { standoff: 15 },
          zerolinecolor: "white",
          automargin: true,
          zerolinewidth: 2,
        },
        scene: {
          xaxis: {
            backgroundcolor: "#E5ECF6",
            gridcolor: "white",
            linecolor: "white",
            showbackground: true,
            ticks: "",
            zerolinecolor: "white",
            gridwidth: 2,
          },
          yaxis: {
            backgroundcolor: "#E5ECF6",
            gridcolor: "white",
            linecolor: "white",
            showbackground: true,
            ticks: "",
            zerolinecolor: "white",
            gridwidth: 2,
          },
          zaxis: {
            backgroundcolor: "#E5ECF6",
            gridcolor: "white",
            linecolor: "white",
            showbackground: true,
            ticks: "",
            zerolinecolor: "white",
            gridwidth: 2,
          },
        },
        shapedefaults: { line: { color: "#2a3f5f" } },
        annotationdefaults: { arrowcolor: "#2a3f5f", arrowhead: 0, arrowwidth: 1 },
        geo: {
          bgcolor: "white",
          landcolor: "#E5ECF6",
          subunitcolor: "white",
          showland: true,
          showlakes: true,
          lakecolor: "white",
        },
        title: { x: 0.05 },
        mapbox: { style: "light" },
      },
    },
  },
};
const secondFakePlot = {
  data: [{ x: [4, 5, 6], y: [7, 8, 9], type: "bar" }],
  layout: {
    title: { text: "Another Figure" },
    template: {
      data: {
        histogram2dcontour: [
          {
            type: "histogram2dcontour",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        choropleth: [{ type: "choropleth", colorbar: { outlinewidth: 0, ticks: "" } }],
        histogram2d: [
          {
            type: "histogram2d",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        heatmap: [
          {
            type: "heatmap",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        heatmapgl: [
          {
            type: "heatmapgl",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        contourcarpet: [{ type: "contourcarpet", colorbar: { outlinewidth: 0, ticks: "" } }],
        contour: [
          {
            type: "contour",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        surface: [
          {
            type: "surface",
            colorbar: { outlinewidth: 0, ticks: "" },
            colorscale: [
              [0.0, "#0d0887"],
              [0.1111111111111111, "#46039f"],
              [0.2222222222222222, "#7201a8"],
              [0.3333333333333333, "#9c179e"],
              [0.4444444444444444, "#bd3786"],
              [0.5555555555555556, "#d8576b"],
              [0.6666666666666666, "#ed7953"],
              [0.7777777777777778, "#fb9f3a"],
              [0.8888888888888888, "#fdca26"],
              [1.0, "#f0f921"],
            ],
          },
        ],
        mesh3d: [{ type: "mesh3d", colorbar: { outlinewidth: 0, ticks: "" } }],
        scatter: [
          { fillpattern: { fillmode: "overlay", size: 10, solidity: 0.2 }, type: "scatter" },
        ],
        parcoords: [{ type: "parcoords", line: { colorbar: { outlinewidth: 0, ticks: "" } } }],
        scatterpolargl: [
          { type: "scatterpolargl", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        bar: [
          {
            error_x: { color: "#2a3f5f" },
            error_y: { color: "#2a3f5f" },
            marker: {
              line: { color: "#E5ECF6", width: 0.5 },
              pattern: { fillmode: "overlay", size: 10, solidity: 0.2 },
            },
            type: "bar",
          },
        ],
        scattergeo: [{ type: "scattergeo", marker: { colorbar: { outlinewidth: 0, ticks: "" } } }],
        scatterpolar: [
          { type: "scatterpolar", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        histogram: [
          {
            marker: { pattern: { fillmode: "overlay", size: 10, solidity: 0.2 } },
            type: "histogram",
          },
        ],
        scattergl: [{ type: "scattergl", marker: { colorbar: { outlinewidth: 0, ticks: "" } } }],
        scatter3d: [
          {
            type: "scatter3d",
            line: { colorbar: { outlinewidth: 0, ticks: "" } },
            marker: { colorbar: { outlinewidth: 0, ticks: "" } },
          },
        ],
        scattermapbox: [
          { type: "scattermapbox", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        scatterternary: [
          { type: "scatterternary", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        scattercarpet: [
          { type: "scattercarpet", marker: { colorbar: { outlinewidth: 0, ticks: "" } } },
        ],
        carpet: [
          {
            aaxis: {
              endlinecolor: "#2a3f5f",
              gridcolor: "white",
              linecolor: "white",
              minorgridcolor: "white",
              startlinecolor: "#2a3f5f",
            },
            baxis: {
              endlinecolor: "#2a3f5f",
              gridcolor: "white",
              linecolor: "white",
              minorgridcolor: "white",
              startlinecolor: "#2a3f5f",
            },
            type: "carpet",
          },
        ],
        table: [
          {
            cells: { fill: { color: "#EBF0F8" }, line: { color: "white" } },
            header: { fill: { color: "#C8D4E3" }, line: { color: "white" } },
            type: "table",
          },
        ],
        barpolar: [
          {
            marker: {
              line: { color: "#E5ECF6", width: 0.5 },
              pattern: { fillmode: "overlay", size: 10, solidity: 0.2 },
            },
            type: "barpolar",
          },
        ],
        pie: [{ automargin: true, type: "pie" }],
      },
      layout: {
        autotypenumbers: "strict",
        colorway: [
          "#636efa",
          "#EF553B",
          "#00cc96",
          "#ab63fa",
          "#FFA15A",
          "#19d3f3",
          "#FF6692",
          "#B6E880",
          "#FF97FF",
          "#FECB52",
        ],
        font: { color: "#2a3f5f" },
        hovermode: "closest",
        hoverlabel: { align: "left" },
        paper_bgcolor: "white",
        plot_bgcolor: "#E5ECF6",
        polar: {
          bgcolor: "#E5ECF6",
          angularaxis: { gridcolor: "white", linecolor: "white", ticks: "" },
          radialaxis: { gridcolor: "white", linecolor: "white", ticks: "" },
        },
        ternary: {
          bgcolor: "#E5ECF6",
          aaxis: { gridcolor: "white", linecolor: "white", ticks: "" },
          baxis: { gridcolor: "white", linecolor: "white", ticks: "" },
          caxis: { gridcolor: "white", linecolor: "white", ticks: "" },
        },
        coloraxis: { colorbar: { outlinewidth: 0, ticks: "" } },
        colorscale: {
          sequential: [
            [0.0, "#0d0887"],
            [0.1111111111111111, "#46039f"],
            [0.2222222222222222, "#7201a8"],
            [0.3333333333333333, "#9c179e"],
            [0.4444444444444444, "#bd3786"],
            [0.5555555555555556, "#d8576b"],
            [0.6666666666666666, "#ed7953"],
            [0.7777777777777778, "#fb9f3a"],
            [0.8888888888888888, "#fdca26"],
            [1.0, "#f0f921"],
          ],
          sequentialminus: [
            [0.0, "#0d0887"],
            [0.1111111111111111, "#46039f"],
            [0.2222222222222222, "#7201a8"],
            [0.3333333333333333, "#9c179e"],
            [0.4444444444444444, "#bd3786"],
            [0.5555555555555556, "#d8576b"],
            [0.6666666666666666, "#ed7953"],
            [0.7777777777777778, "#fb9f3a"],
            [0.8888888888888888, "#fdca26"],
            [1.0, "#f0f921"],
          ],
          diverging: [
            [0, "#8e0152"],
            [0.1, "#c51b7d"],
            [0.2, "#de77ae"],
            [0.3, "#f1b6da"],
            [0.4, "#fde0ef"],
            [0.5, "#f7f7f7"],
            [0.6, "#e6f5d0"],
            [0.7, "#b8e186"],
            [0.8, "#7fbc41"],
            [0.9, "#4d9221"],
            [1, "#276419"],
          ],
        },
        xaxis: {
          gridcolor: "white",
          linecolor: "white",
          ticks: "",
          title: { standoff: 15 },
          zerolinecolor: "white",
          automargin: true,
          zerolinewidth: 2,
        },
        yaxis: {
          gridcolor: "white",
          linecolor: "white",
          ticks: "",
          title: { standoff: 15 },
          zerolinecolor: "white",
          automargin: true,
          zerolinewidth: 2,
        },
        scene: {
          xaxis: {
            backgroundcolor: "#E5ECF6",
            gridcolor: "white",
            linecolor: "white",
            showbackground: true,
            ticks: "",
            zerolinecolor: "white",
            gridwidth: 2,
          },
          yaxis: {
            backgroundcolor: "#E5ECF6",
            gridcolor: "white",
            linecolor: "white",
            showbackground: true,
            ticks: "",
            zerolinecolor: "white",
            gridwidth: 2,
          },
          zaxis: {
            backgroundcolor: "#E5ECF6",
            gridcolor: "white",
            linecolor: "white",
            showbackground: true,
            ticks: "",
            zerolinecolor: "white",
            gridwidth: 2,
          },
        },
        shapedefaults: { line: { color: "#2a3f5f" } },
        annotationdefaults: { arrowcolor: "#2a3f5f", arrowhead: 0, arrowwidth: 1 },
        geo: {
          bgcolor: "white",
          landcolor: "#E5ECF6",
          subunitcolor: "white",
          showland: true,
          showlakes: true,
          lakecolor: "white",
        },
        title: { x: 0.05 },
        mapbox: { style: "light" },
      },
    },
  },
};
const plots: PlotDto[] = [
  { name: "plot 1", value: fakePlot },
  {
    name: "plot 2",
    value: secondFakePlot,
  },
];

const sections: DetailSectionDto[] = [
  {
    title: "Model",
    icon: "icon-square-cursor",
    items: [
      {
        name: "Estimator parameters:",
        description: "Core model configuration used for training",
        value: "`RandomForestClassifier` *100* trees, max_depth *10*",
      },
      {
        name: "Estimator parameters:",
        description: "Core model configuration used for training",
        value: "`RandomForestClassifier` *100* trees, max_depth *10*",
      },
      {
        name: "Estimator parameters:",
        description: "Core model configuration used for training",
        value: "`RandomForestClassifier` *100* trees, max_depth *10*",
      },
    ],
  },
  {
    title: "bla bla",
    icon: "icon-text",
    items: [
      {
        name: "Estimator parameters:",
        description: "Core model configuration used for training",
        value: "`RandomForestClassifier` *100* trees, max_depth *10*",
      },
      {
        name: "Estimator parameters:",
        description: "Core model configuration used for training",
        value: "`RandomForestClassifier` *100* trees, max_depth *10*",
      },
      {
        name: "Estimator parameters:",
        description: "Core model configuration used for training",
        value: "`RandomForestClassifier` *100* trees, max_depth *10*",
      },
    ],
  },
];

const toggleModel = ref(true);

const richTextEditor = useTemplateRef<InstanceType<typeof RichTextEditor>>("richTextEditor");
const richText = ref(
  "I don’t ‘need’ to drink. I can quit anytime I want! Hello, little man. I will destroy you! Kids have names? You won’t have time for sleeping, soldier, not with all the bed making you’ll be doing. When the lights go out, it’s nobody’s business what goes on between two consenting adults."
);
</script>

<template>
  <main>
    <h1>Components library</h1>
    <TabPanel>
      <TabPanelContent name="markdown">
        <Simplebar class="markdown-list-container">
          <MarkdownWidget :source="markdownString" />
        </Simplebar>
      </TabPanelContent>
      <TabPanelContent name="vega">
        <VegaWidget :spec="spec as VisualizationSpec" :theme="themesStore.currentTheme" />
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
          @item-action="
            (action: string, itemName: string) => (lastAction = `${action} ${itemName}`)
          "
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
          <div>icon-moon <i class="icon icon-moon"></i></div>
          <div>icon-sun <i class="icon icon-sun"></i></div>
          <div>icon-ascending-arrow <i class="icon icon-ascending-arrow"></i></div>
          <div>icon-descending-arrow <i class="icon icon-descending-arrow"></i></div>
          <div>icon-bold <i class="icon icon-bold"></i></div>
          <div>icon-italic <i class="icon icon-italic"></i></div>
          <div>icon-bullets <i class="icon icon-bullets"></i></div>
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
                <div data-drag-image-selector :style="{ backgroundColor: color }">ID: {{ id }}</div>
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
        <CrossValidationReport
          :scalar-results="results.scalarResults"
          :tabular-results="results.tabularResults"
          :plots="plots"
          :sections="sections"
        />
      </TabPanelContent>
      <TabPanelContent name="toggle" class="toggles">
        <div>
          <SlideToggle v-model:is-toggled="toggleModel" />
        </div>
        <div>
          <SlideToggle v-model:is-toggled="toggleModel" />
        </div>
        <div>toggles are {{ toggleModel }}</div>
      </TabPanelContent>
      <TabPanelContent name="rich edit" class="rich">
        <div class="actions">
          <SimpleButton :is-primary="false" label="bold" @click="richTextEditor?.markBold()" />
          <SimpleButton :is-primary="false" label="italic" @click="richTextEditor?.markItalic()" />
          <SimpleButton :is-primary="false" label="list" @click="richTextEditor?.markList()" />
        </div>
        <div style="height: 200px">
          <RichTextEditor ref="richTextEditor" v-model:value="richText" />
        </div>
        <MarkdownWidget :source="richText" />
      </TabPanelContent>
    </TabPanel>
  </main>
</template>

<style scoped>
main {
  padding: 0 5vw;
  color: var(--color-text-primary);

  & h1 {
    margin: var(--spacing-8) 0;
  }
}

.markdown-list-container {
  max-height: 70dvh;
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
  max-height: 60vh;
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

.toggles {
  display: flex;
  flex-direction: column;
  padding: var(--spacing-8);
  gap: var(--spacing-8);
}

.rich {
  padding: var(--spacing-8);

  & .actions {
    display: flex;
    flex-direction: row;
    justify-content: flex-end;
    padding-bottom: var(--spacing-8);
    gap: var(--spacing-8);
  }
}
</style>
