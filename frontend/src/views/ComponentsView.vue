<script setup lang="ts">
import markdownString from "../assets/fixtures/markdown.md?raw";
import MarkdownWidget from "../components/MarkdownWidget.vue";
</script>

<template>
  <main>
    <h1>Components library</h1>
    <MarkdownWidget :source="markdownString" />
  </main>
</template>
