<script setup lang="ts">
import { nextTick, ref, useTemplateRef } from "vue";

import MarkdownWidget from "@/components/MarkdownWidget.vue";
import MediaWidgetSelector from "@/components/MediaWidgetSelector.vue";
import ActivityFeedCardHeader from "@/views/activity/ActivityFeedCardHeader.vue";
import ItemNoteEditor from "@/views/activity/ItemNoteEditor.vue";
import type { ActivityPresentableItem } from "@/views/activity/activity";

const props = defineProps<{ item: ActivityPresentableItem }>();

const isAnnotating = ref(false);
const editor = useTemplateRef<InstanceType<typeof ItemNoteEditor>>("editor");

function annotateItem() {
  isAnnotating.value = true;
  nextTick(() => {
    editor.value?.focus();
  });
}

function finishEdition() {
  isAnnotating.value = false;
}
</script>

<template>
  <div class="activity-feed-item" :class="{ 'has-note': item.note && item.note.length > 1 }">
    <ActivityFeedCardHeader
      :icon="props.item.icon"
      :datetime="props.item.updatedAt"
      :name="props.item.name"
      :version="props.item.version"
      @annotate="annotateItem()"
    />
    <ItemNoteEditor
      v-if="isAnnotating"
      ref="editor"
      :name="props.item.name"
      :version="props.item.version"
      :note="props.item.note ?? ''"
      @edition-end="finishEdition"
    />
    <MarkdownWidget
      v-if="props.item.note && props.item.note.length > 0 && !isAnnotating"
      :source="props.item.note"
      @click.stop="annotateItem()"
    />
    <div class="item-wrapper">
      <MediaWidgetSelector :item="item" />
    </div>
  </div>
</template>

<style scoped>
.activity-feed-item {
  &.has-note .item-wrapper {
    margin-top: var(--spacing-4);
  }
}
</style>
