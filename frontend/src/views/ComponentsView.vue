<script setup lang="ts">
import type { VisualizationSpec } from "vega-embed";

import DataFrameWidget from "../components/DataFrameWidget.vue";
import MarkdownWidget from "../components/MarkdownWidget.vue";
import Tabs from "../components/TabsWidget.vue";
import TabsItem from "../components/TabsWidgetItem.vue";
import VegaWidget from "../components/VegaWidget.vue";

import markdownString from "../assets/fixtures/markdown.md?raw";
import spec from "../assets/fixtures/vega.json";
</script>

<template>
  <main>
    <h1>Components library</h1>
    <Tabs :tab-names="['markdown', 'vega', 'DataFrame']">
      <TabsItem :value="0">
        <MarkdownWidget :source="markdownString" />
      </TabsItem>
      <TabsItem :value="1">
        <VegaWidget :spec="spec as VisualizationSpec" />
      </TabsItem>
      <TabsItem :value="2">
        <DataFrameWidget
          :headers="['plip', 'plop']"
          :values="[
            [1, 2],
            [2, 3],
          ]"
        />
      </TabsItem>
    </Tabs>
  </main>
</template>
