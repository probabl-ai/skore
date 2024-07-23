<script setup lang="ts">
import "highlight.js/styles/atom-one-light.min.css";
import MarkdownIt from "markdown-it";
import { full as emoji } from "markdown-it-emoji";
import highlightjs from "markdown-it-highlightjs";
import sub from "markdown-it-sub";
import sup from "markdown-it-sup";
import { computed } from "vue";

const props = defineProps<{ source: string }>();
const renderer = MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
})
  .use(emoji)
  .use(sub)
  .use(sup)
  .use(highlightjs, { inline: true });

const html = computed(() => {
  return renderer.render(props.source);
});
</script>

<template>
  <div class="markdown" v-html="html"></div>
</template>
