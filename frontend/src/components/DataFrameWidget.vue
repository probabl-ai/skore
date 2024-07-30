<script setup lang="ts">
import "datatables.net-dt/css/dataTables.dataTables.min.css";

import DataTablesCore from "datatables.net";
import DataTable from "datatables.net-vue3";

DataTable.use(DataTablesCore);

export interface DataFrameWidgetProps {
  columns: string[];
  data: any[][];
}

const props = defineProps<DataFrameWidgetProps>();
</script>

<template>
  <DataTable :data="props.data">
    <thead>
      <tr>
        <th v-for="(name, index) in props.columns" :key="index">{{ name }}</th>
      </tr>
    </thead>
  </DataTable>
</template>
