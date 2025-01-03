<script setup lang="ts">
import katex from "@vscode/markdown-it-katex";
import hljs from "highlight.js/lib/core";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import xml from "highlight.js/lib/languages/xml";
import "katex/dist/katex.min.css";
import MarkdownIt from "markdown-it";
import { full as emoji } from "markdown-it-emoji";
import highlightjs from "markdown-it-highlightjs/core";
import sub from "markdown-it-sub";
import sup from "markdown-it-sup";
import { computed } from "vue";

import { isString } from "@/services/utils";

hljs.registerLanguage("python", python);
hljs.registerLanguage("sql", sql);
hljs.registerLanguage("json", json);
hljs.registerLanguage("xml", xml);

const props = defineProps<{ source: any }>();
const renderer = MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
})
  .use(emoji)
  .use(sub)
  .use(sup)
  .use(katex)
  .use(highlightjs, { inline: true, hljs });

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

<style>
.markdown {
  color: var(--color-text-primary);
}
</style>
