<script setup lang="ts">
import markdownString from "../assets/fixtures/markdown.md?raw";
import MarkdownWidget from "../components/MarkdownWidget.vue";
import Tabs from "../components/TabsWidget.vue";
import TabsItem from "../components/TabsWidgetItem.vue";
import VegaWidget from "../components/VegaWidget.vue";

import spec from "../assets/fixtures/vega.json";
</script>

<template>
  <main>
    <h1>Components library</h1>
    <Tabs :tab-names="['markdown', 'vega']">
      <TabsItem :value="0">
        <MarkdownWidget :source="markdownString" />
      </TabsItem>
      <TabsItem :value="1">
        <VegaWidget :spec="spec" />
      </TabsItem>
    </Tabs>
  </main>
</template>
