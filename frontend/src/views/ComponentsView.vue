<script setup lang="ts">
import markdownString from "../assets/fixtures/markdown.md?raw";
import MarkdownWidget from "../components/MarkdownWidget.vue";
import Tabs from "../components/TabsWidget.vue";
import TabsItem from "../components/TabsWidgetItem.vue";
</script>

<template>
  <main>
    <h1>Components library</h1>
    <Tabs :tab-names="['markdown', 'vega']">
      <TabsItem :value="0">
        <MarkdownWidget :source="markdownString" />
      </TabsItem>
      <TabsItem :value="1">
        <div>vega</div>
      </TabsItem>
    </Tabs>
  </main>
</template>
