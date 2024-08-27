<script setup lang="ts">
import { isString } from "@/services/utils";
import MarkdownIt from "markdown-it";
import { full as emoji } from "markdown-it-emoji";
import highlightjs from "markdown-it-highlightjs";
import sub from "markdown-it-sub";
import sup from "markdown-it-sup";
import { computed } from "vue";

const props = defineProps<{ source: any }>();
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
  let s = props.source;
  if (!isString(s)) {
    s = JSON.stringify(s, null, 2);
    s = "```json\n" + s + "\n```";
  }
  return renderer.render(s);
});
</script>

<template>
  <div class="markdown" v-html="html"></div>
</template>
