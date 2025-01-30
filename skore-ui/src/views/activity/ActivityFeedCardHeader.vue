<script setup lang="ts">
import { format } from "date-fns";
import { computed } from "vue";

const props = defineProps<{
  icon: string;
  name: string;
  version: number;
  datetime: Date;
}>();

const date = computed(() => format(props.datetime, "yyyy/MM/dd"));
const time = computed(() => format(props.datetime, "HH:mm:ss"));
</script>

<template>
  <div class="activity-feed-card-header">
    <div class="name">
      <i class="icon" :class="props.icon" /> {{ props.name }}
      <span v-if="version > 0" class="version">#{{ props.version }}</span>
    </div>
    <div class="datetime">{{ date }} <span class="at">at</span> {{ time }}</div>
  </div>
</template>

<style scoped>
.activity-feed-card-header {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  margin-bottom: var(--spacing-4);

  & .name {
    color: var(--color-text-primary);
    font-size: var(--font-size-xs);

    & .icon {
      color: var(--color-icon-tertiary);
      vertical-align: middle;
    }

    & .version {
      color: var(--color-text-secondary);
      font-size: var(--font-size-xxs);
    }
  }

  & .datetime {
    color: var(--color-text-primary);
    font-size: var(--font-size-xs);

    & .at {
      color: var(--color-text-secondary);
    }
  }
}
</style>
