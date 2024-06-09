package com.example.stylosense.presentations.screen.cart_screen

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.constraintlayout.compose.ConstraintLayout
import com.example.stylosense.R
import com.example.stylosense.presentations.commons.widgets.CustomDefaultBtn
import com.example.stylosense.presentations.commons.widgets.DefaultBackArrow


@Preview(showBackground = true)
@Composable
fun CartScreen() {
    var itemDrag by remember { mutableStateOf(0f) }


    ConstraintLayout(modifier = Modifier.fillMaxSize(1f)) {
        val (topBar, product, checkout) = createRefs()

        Row(
            modifier = Modifier
                .padding(top = 15.dp, start = 15.dp, end = 15.dp)
                .fillMaxWidth()
                .constrainAs(topBar) {
                    top.linkTo(parent.top)
                    start.linkTo(parent.start)
                    end.linkTo(parent.end)
                },
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(modifier = Modifier.weight(0.5f)) {
                DefaultBackArrow {

                }
            }
            Box(modifier = Modifier.weight(0.7f)) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
//                    Text(
//                        text = "Your Cart",
//                        color = MaterialTheme.colorScheme.primary,
//                        fontSize = 18.sp,
//                        fontWeight = FontWeight.Bold
//                    )
//                    Text(
//                        text = "4 items",
//                        textAlign = TextAlign.Center,
//                        color = MaterialTheme.colorScheme.primary,
//                    )

                }
            }

        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .constrainAs(product) {
                    top.linkTo(topBar.bottom)
                    start.linkTo(parent.start)
                    end.linkTo(parent.end)
                }
                .wrapContentHeight()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(15.dp)
                    .pointerInput(Unit) {
                        detectVerticalDragGestures { change, dragAmount ->
                            itemDrag = dragAmount
                        }
                    },
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.ps4_console_white_1),
                    contentDescription = null,
                    modifier = Modifier
                        .size(80.dp)
                        .background(Color(0x8DB3B0B0), shape = RoundedCornerShape(10.dp))
                        .padding(10.dp)
                        .clip(RoundedCornerShape(10.dp))
                )
                Column() {
//                    Text(
//                        text = "Wireless Controller for PS4™",
//                        fontWeight = FontWeight(700),
//                        fontSize = 16.sp,
//
//                        )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row() {
//                        Text(
//                            text = "$79.99",
//                            color = MaterialTheme.colors.PrimaryColor,
//                            fontWeight = FontWeight.Bold
//                        )
//                        Text(text = "  x1", color = MaterialTheme.colorScheme.primary)
                    }
                }
            }







            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(15.dp)
                    .pointerInput(Unit) {
                        detectVerticalDragGestures { change, dragAmount ->
                            itemDrag = dragAmount
                        }
                    },
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.shoes2),
                    contentDescription = null,
                    modifier = Modifier
                        .size(80.dp)
                        .background(Color(0x8DB3B0B0), shape = RoundedCornerShape(10.dp))
                        .padding(10.dp)
                        .clip(RoundedCornerShape(10.dp))
                )
                Column() {
//                    Text(
//                        text = "High Quality Sport Shoes",
//                        fontWeight = FontWeight(700),
//                        fontSize = 16.sp,
//
//                        )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row() {
//                        Text(
//                            text = "$100.25",
//                            color = MaterialTheme.colorScheme.primary,
//                            fontWeight = FontWeight.Bold
//                        )
//                        Text(text = "  x1", color = MaterialTheme.colorScheme.primary)
                    }
                }
            }



            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(15.dp)
                    .pointerInput(Unit) {
                        detectVerticalDragGestures { change, dragAmount ->
                            itemDrag = dragAmount
                        }
                    },
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.image_popular_product_2),
                    contentDescription = null,
                    modifier = Modifier
                        .size(80.dp)
                        .background(Color(0x8DB3B0B0), shape = RoundedCornerShape(10.dp))
                        .padding(10.dp)
                        .clip(RoundedCornerShape(10.dp))
                )
                Column() {
//                    Text(
//                        text = "Nike Sport White - Man Pant",
//                        fontWeight = FontWeight(700),
//                        fontSize = 16.sp,
//
//                        )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row() {
//                        Text(
//                            text = "$49.99",
//                            color = MaterialTheme.colorScheme.primary,
//                            fontWeight = FontWeight.Bold
//                        )
//                        Text(text = "  x1", color = MaterialTheme.colorScheme.primary)
                    }
                }
            }



            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(15.dp)
                    .pointerInput(Unit) {
                        detectVerticalDragGestures { change, dragAmount ->
                            itemDrag = dragAmount
                        }
                    },
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.glove),
                    contentDescription = null,
                    modifier = Modifier
                        .size(80.dp)
                        .background(Color(0x8DB3B0B0), shape = RoundedCornerShape(10.dp))
                        .padding(10.dp)
                        .clip(RoundedCornerShape(10.dp))
                )
                Column() {
//                    Text(
//                        text = "Gloves XC Omega - Polygon",
//                        fontWeight = FontWeight(700),
//                        fontSize = 16.sp,
//
//                        )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row() {
//                        Text(
//                            text = "$36.55",
//                            color = MaterialTheme.colors.PrimaryColor,
//                            fontWeight = FontWeight.Bold
//                        )
//                        Text(text = "  x1", color = MaterialTheme.colorScheme.primary)
                    }
                }
            }


        }




        Column(
            modifier = Modifier
                .wrapContentHeight()
                .constrainAs(checkout) {
                    bottom.linkTo(parent.bottom)
                    start.linkTo(parent.start)
                    end.linkTo(parent.end)
                }
                .background(
                    color = MaterialTheme.colorScheme.primary,
                    shape = RoundedCornerShape(topStart = 15.dp, topEnd = 15.dp)
                )
                .clip(RoundedCornerShape(topStart = 15.dp, topEnd = 15.dp))
                .padding(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    painter = painterResource(id = R.drawable.receipt),
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier
                        .size(45.dp)
                        .background(Color(0x8DB3B0B0), shape = RoundedCornerShape(15.dp))
                        .padding(10.dp)
                        .clip(RoundedCornerShape(15.dp))
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(5.dp)
                ) {
//                    Text("Add vouture Code")
                    Icon(
                        painter = painterResource(id = R.drawable.arrow_right),
                        contentDescription = null,
                    )
                }
            }
            //btn
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column() {
//                    Text(text = "Total")
//                    Text(
//                        text = "$266.78",
//                        fontSize = 18.sp,
//                        fontWeight = FontWeight.Bold,
//                        color = MaterialTheme.colors.PrimaryColor
//                    )

                }
                Box(
                    modifier = Modifier
                        .width(150.dp)
                ) {
                    CustomDefaultBtn(shapeSize = 15f, btnText = "Check Out") {

                    }
                }

            }


        }

    }
}